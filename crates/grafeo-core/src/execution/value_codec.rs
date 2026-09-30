//! Lossless, checked binary encoding for execution [`Value`] rows.
//!
//! This module is built independently of disk spilling so resident and spilled
//! execution can converge on one measured representation without changing
//! query semantics. The established spill-v1 bytes remain the compatibility
//! format. The codec is designed for:
//! - Minimal overhead (no schema, direct binary encoding)
//! - Fast serialization/deserialization
//! - Compact representation

use arcstr::ArcStr;
use grafeo_common::memory::buffer::MemoryGrantError;
use grafeo_common::types::Value;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::Arc;

#[cfg(test)]
std::thread_local! {
    static SEMANTIC_KEY_TRAVERSAL_VISITS: std::cell::Cell<(usize, usize)> = const {
        std::cell::Cell::new((0, 0))
    };
}

#[cfg(test)]
pub(crate) fn take_semantic_key_traversal_visits() -> (usize, usize) {
    SEMANTIC_KEY_TRAVERSAL_VISITS.with(|visits| visits.replace((0, 0)))
}

// Type tags for Value variants
const TAG_NULL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_INT64: u8 = 2;
const TAG_FLOAT64: u8 = 3;
const TAG_STRING: u8 = 4;
const TAG_BYTES: u8 = 5;
const TAG_TIMESTAMP: u8 = 6;
const TAG_LIST: u8 = 7;
const TAG_MAP: u8 = 8;
const TAG_VECTOR: u8 = 9;
const TAG_DATE: u8 = 10;
const TAG_TIME: u8 = 11;
const TAG_DURATION: u8 = 12;
const TAG_PATH: u8 = 13;
const TAG_ZONED_DATETIME: u8 = 14;
const TAG_GCOUNTER: u8 = 15;
const TAG_PNCOUNTER: u8 = 16;
const TAG_RDF_LITERAL: u8 = 17;
const TAG_TIME_EXACT: u8 = 18;

// Explicitly bounded callers use conservative resource defaults. Legacy
// no-context wrappers retain the pre-existing execution capability envelope by
// using the platform's checked allocation maximum, while all paths keep the
// hard recursion-safety ceiling and incremental/fallible allocation.
const DEFAULT_MAX_CODEC_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_CODEC_ITEMS: usize = 1_000_000;
const DEFAULT_MAX_CODEC_ROW_COLUMNS: usize = 65_536;
const DEFAULT_MAX_CODEC_DEPTH: usize = 128;
const FORMAT_MAX_CODEC_ALLOCATION: usize = isize::MAX as usize;
const DECODE_GROWTH_CHUNK: usize = 64 * 1024;

/// Safety limits for one value or row codec operation.
///
/// `CodecLimits::default()` is conservative for callers that explicitly choose
/// bounded operation. The legacy no-context codec wrappers instead use checked
/// platform allocation-format maxima until query execution passes real resource
/// grants to the limit-aware entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecLimits {
    max_bytes: usize,
    max_items: usize,
    max_row_columns: usize,
    max_depth: usize,
}

impl CodecLimits {
    /// Creates codec limits for one value or row operation.
    ///
    /// `max_depth` is clamped to the format-safety ceiling of 128. Byte, item,
    /// and column limits may be raised by a caller with a larger resource grant,
    /// up to the platform allocation-format maximum.
    #[must_use]
    pub const fn new(
        max_bytes: usize,
        max_items: usize,
        max_row_columns: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            max_bytes: clamp_allocation_limit(max_bytes),
            max_items: clamp_allocation_limit(max_items),
            max_row_columns: clamp_allocation_limit(max_row_columns),
            max_depth: if max_depth > DEFAULT_MAX_CODEC_DEPTH {
                DEFAULT_MAX_CODEC_DEPTH
            } else {
                max_depth
            },
        }
    }

    /// Returns the checked platform allocation-format maxima.
    ///
    /// This is a compatibility ceiling, not a production memory grant.
    #[must_use]
    pub const fn format_max() -> Self {
        Self::new(
            FORMAT_MAX_CODEC_ALLOCATION,
            FORMAT_MAX_CODEC_ALLOCATION,
            FORMAT_MAX_CODEC_ALLOCATION,
            DEFAULT_MAX_CODEC_DEPTH,
        )
    }

    /// Tightens physical item/column count bounds to bytes present in one
    /// already-framed payload while preserving the caller's decoded-byte grant.
    #[must_use]
    pub const fn bounded_to_payload(self, payload_bytes: usize) -> Self {
        Self {
            // Decoded resident representation can legitimately exceed compact
            // wire bytes (for example, a list of Null values). Keep the
            // caller's allocation grant. The standard framed-row decoder
            // separately exposes its remaining wire bytes so string/byte
            // lengths are rejected before growth.
            max_bytes: self.max_bytes,
            max_items: if self.max_items < payload_bytes {
                self.max_items
            } else {
                payload_bytes
            },
            max_row_columns: if self.max_row_columns < payload_bytes {
                self.max_row_columns
            } else {
                payload_bytes
            },
            max_depth: self.max_depth,
        }
    }

    /// Returns the maximum decoded bytes accepted by one codec operation.
    #[must_use]
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }

    /// Returns the maximum decoded collection items accepted by one codec operation.
    #[must_use]
    pub const fn max_items(self) -> usize {
        self.max_items
    }
}

const fn clamp_allocation_limit(limit: usize) -> usize {
    if limit > FORMAT_MAX_CODEC_ALLOCATION {
        FORMAT_MAX_CODEC_ALLOCATION
    } else {
        limit
    }
}

impl Default for CodecLimits {
    fn default() -> Self {
        Self::new(
            DEFAULT_MAX_CODEC_BYTES,
            DEFAULT_MAX_CODEC_ITEMS,
            DEFAULT_MAX_CODEC_ROW_COLUMNS,
            DEFAULT_MAX_CODEC_DEPTH,
        )
    }
}

struct CodecBudget {
    limits: CodecLimits,
    remaining_bytes: usize,
    remaining_items: usize,
}

/// Diagnostic primary used by the grant-qualified framed decoder.
///
/// The exact-slice qualified entry point can reach only allocation-free
/// variants: it never converts this value into [`std::io::Error`] or formats an
/// owned message. Compatibility entry points perform that conversion only
/// after the qualified decode has returned. Numeric context and static field
/// names therefore remain inspectable without constructing a second heap
/// payload on malformed-input paths. The private `Io` variant exists solely so
/// unframed compatibility readers can preserve their original I/O error.
enum QualifiedCodecPrimary {
    Io(std::io::Error),
    TruncatedFrame {
        requested: usize,
        remaining: usize,
    },
    #[cfg(feature = "spill")]
    MissingFrameBoundary,
    FramedLengthExceeds {
        description: &'static str,
        length: usize,
        remaining: usize,
    },
    LengthNotAddressable {
        description: &'static str,
        encoded: u64,
    },
    LengthExceeds {
        description: &'static str,
        length: usize,
        maximum: usize,
    },
    BudgetSizeOverflow {
        description: &'static str,
        count: usize,
        item_size: usize,
    },
    ByteBudgetExceeded {
        description: &'static str,
        requested: usize,
        remaining: usize,
        maximum: usize,
    },
    ItemLimitExceeded {
        description: &'static str,
        count: usize,
        maximum: usize,
    },
    ItemBudgetExceeded {
        description: &'static str,
        requested: usize,
        remaining: usize,
        maximum: usize,
    },
    DepthExceeded {
        depth: usize,
        maximum: usize,
    },
    InvalidBoolean(u8),
    InvalidUtf8 {
        description: &'static str,
        error: std::str::Utf8Error,
    },
    Allocation {
        description: &'static str,
        error: std::collections::TryReserveError,
    },
    SharedAllocation {
        description: &'static str,
    },
    PayloadRetainedOverflow {
        description: &'static str,
    },
    #[cfg(feature = "spill")]
    PayloadTrackingDisabled,
    InvalidTimeNanos(u64),
    InvalidTimeOffsetPresence(u8),
    MapKeysNotIncreasing,
    UnknownValueTag(u8),
    InvalidOptionalStringPresence {
        description: &'static str,
        value: u8,
    },
    DuplicateCounterKey,
    RowColumnMismatch {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "spill")]
    FramedRowTooShort {
        actual: usize,
    },
    #[cfg(feature = "spill")]
    TrailingFrameBytes {
        remaining: usize,
    },
    ConstructionEnvelope(MemoryGrantError),
    #[cfg(feature = "spill")]
    RetainedBeyondEnvelope {
        observed: usize,
        envelope: usize,
    },
}

/// Opaque qualified diagnostic; its owned primary cannot be detached.
pub(crate) struct QualifiedCodecError {
    primary: QualifiedCodecPrimary,
}

/// Copy-only classification exposed by a qualified codec diagnostic.
#[cfg(all(test, feature = "spill"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QualifiedCodecErrorClass {
    TruncatedFrame {
        requested: usize,
        remaining: usize,
    },
    FramedLengthExceeds {
        description: &'static str,
        length: usize,
        remaining: usize,
    },
    Other(std::io::ErrorKind),
}

impl std::fmt::Debug for QualifiedCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QualifiedCodecError")
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

impl QualifiedCodecError {
    pub(crate) fn kind(&self) -> std::io::ErrorKind {
        match &self.primary {
            QualifiedCodecPrimary::Io(error) => error.kind(),
            QualifiedCodecPrimary::Allocation { .. }
            | QualifiedCodecPrimary::SharedAllocation { .. }
            | QualifiedCodecPrimary::PayloadRetainedOverflow { .. }
            | QualifiedCodecPrimary::ConstructionEnvelope(_) => std::io::ErrorKind::OutOfMemory,
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::PayloadTrackingDisabled => std::io::ErrorKind::Other,
            QualifiedCodecPrimary::TruncatedFrame { .. }
            | QualifiedCodecPrimary::FramedLengthExceeds { .. }
            | QualifiedCodecPrimary::LengthNotAddressable { .. }
            | QualifiedCodecPrimary::LengthExceeds { .. }
            | QualifiedCodecPrimary::BudgetSizeOverflow { .. }
            | QualifiedCodecPrimary::ByteBudgetExceeded { .. }
            | QualifiedCodecPrimary::ItemLimitExceeded { .. }
            | QualifiedCodecPrimary::ItemBudgetExceeded { .. }
            | QualifiedCodecPrimary::DepthExceeded { .. }
            | QualifiedCodecPrimary::InvalidBoolean(_)
            | QualifiedCodecPrimary::InvalidUtf8 { .. }
            | QualifiedCodecPrimary::InvalidTimeNanos(_)
            | QualifiedCodecPrimary::InvalidTimeOffsetPresence(_)
            | QualifiedCodecPrimary::MapKeysNotIncreasing
            | QualifiedCodecPrimary::UnknownValueTag(_)
            | QualifiedCodecPrimary::InvalidOptionalStringPresence { .. }
            | QualifiedCodecPrimary::DuplicateCounterKey
            | QualifiedCodecPrimary::RowColumnMismatch { .. } => std::io::ErrorKind::InvalidData,
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::MissingFrameBoundary
            | QualifiedCodecPrimary::FramedRowTooShort { .. }
            | QualifiedCodecPrimary::TrailingFrameBytes { .. }
            | QualifiedCodecPrimary::RetainedBeyondEnvelope { .. } => {
                std::io::ErrorKind::InvalidData
            }
        }
    }

    #[cfg(all(test, feature = "spill"))]
    pub(crate) fn class(&self) -> QualifiedCodecErrorClass {
        match &self.primary {
            QualifiedCodecPrimary::TruncatedFrame {
                requested,
                remaining,
            } => QualifiedCodecErrorClass::TruncatedFrame {
                requested: *requested,
                remaining: *remaining,
            },
            QualifiedCodecPrimary::FramedLengthExceeds {
                description,
                length,
                remaining,
            } => QualifiedCodecErrorClass::FramedLengthExceeds {
                description,
                length: *length,
                remaining: *remaining,
            },
            _ => QualifiedCodecErrorClass::Other(self.kind()),
        }
    }

    fn into_io(self) -> std::io::Error {
        match self.primary {
            QualifiedCodecPrimary::Io(error) => error,
            primary => {
                let error = Self { primary };
                std::io::Error::new(error.kind(), error)
            }
        }
    }
}

impl From<QualifiedCodecPrimary> for QualifiedCodecError {
    fn from(primary: QualifiedCodecPrimary) -> Self {
        Self { primary }
    }
}

impl std::fmt::Display for QualifiedCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.primary {
            QualifiedCodecPrimary::Io(error) => write!(
                formatter,
                "qualified codec I/O failure ({:?})",
                error.kind()
            ),
            QualifiedCodecPrimary::TruncatedFrame {
                requested,
                remaining,
            } => write!(
                formatter,
                "truncated framed row payload: requested {requested} bytes with {remaining} remaining"
            ),
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::MissingFrameBoundary => {
                formatter.write_str("framed decode reader has no exact boundary")
            }
            QualifiedCodecPrimary::FramedLengthExceeds {
                description,
                length,
                remaining,
            } => write!(
                formatter,
                "{description} length {length} exceeds {remaining} remaining framed payload bytes"
            ),
            QualifiedCodecPrimary::LengthNotAddressable {
                description,
                encoded,
            } => write!(
                formatter,
                "{description} length {encoded} does not fit this platform"
            ),
            QualifiedCodecPrimary::LengthExceeds {
                description,
                length,
                maximum,
            } => write!(
                formatter,
                "{description} length {length} exceeds maximum {maximum}"
            ),
            QualifiedCodecPrimary::BudgetSizeOverflow {
                description,
                count,
                item_size,
            } => write!(
                formatter,
                "{description} allocation size overflow for {count} items of {item_size} bytes"
            ),
            QualifiedCodecPrimary::ByteBudgetExceeded {
                description,
                requested,
                remaining,
                maximum,
            } => write!(
                formatter,
                "{description} requests {requested} bytes with {remaining} remaining in the cumulative {maximum}-byte codec budget"
            ),
            QualifiedCodecPrimary::ItemLimitExceeded {
                description,
                count,
                maximum,
            } => write!(
                formatter,
                "{description} count {count} exceeds maximum {maximum} (codec item limit)"
            ),
            QualifiedCodecPrimary::ItemBudgetExceeded {
                description,
                requested,
                remaining,
                maximum,
            } => write!(
                formatter,
                "{description} requests {requested} items with {remaining} remaining in the cumulative {maximum}-item codec budget"
            ),
            QualifiedCodecPrimary::DepthExceeded { depth, maximum } => {
                write!(
                    formatter,
                    "value nesting depth {depth} exceeds maximum {maximum}"
                )
            }
            QualifiedCodecPrimary::InvalidBoolean(value) => {
                write!(formatter, "invalid boolean payload {value}")
            }
            QualifiedCodecPrimary::InvalidUtf8 { description, error } => {
                write!(formatter, "invalid UTF-8 in decoded {description}: {error}")
            }
            QualifiedCodecPrimary::Allocation { description, error } => {
                write!(formatter, "failed to reserve {description}: {error}")
            }
            QualifiedCodecPrimary::SharedAllocation { description } => {
                write!(formatter, "failed to allocate decoded {description}")
            }
            QualifiedCodecPrimary::PayloadRetainedOverflow { description } => write!(
                formatter,
                "decoded {description} retained size exceeds the platform address space"
            ),
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::PayloadTrackingDisabled => {
                formatter.write_str("decoded payload receipt tracking was not enabled")
            }
            QualifiedCodecPrimary::InvalidTimeNanos(value) => {
                write!(formatter, "invalid time nanos {value}")
            }
            QualifiedCodecPrimary::InvalidTimeOffsetPresence(value) => {
                write!(formatter, "invalid time offset presence flag {value}")
            }
            QualifiedCodecPrimary::MapKeysNotIncreasing => {
                formatter.write_str("map keys are duplicate or non-increasing")
            }
            QualifiedCodecPrimary::UnknownValueTag(value) => {
                write!(formatter, "Unknown value tag: {value}")
            }
            QualifiedCodecPrimary::InvalidOptionalStringPresence { description, value } => {
                write!(formatter, "invalid {description} presence flag {value}")
            }
            QualifiedCodecPrimary::DuplicateCounterKey => {
                formatter.write_str("duplicate counter key")
            }
            QualifiedCodecPrimary::RowColumnMismatch { expected, actual } => {
                write!(
                    formatter,
                    "Column count mismatch: expected {expected}, got {actual}"
                )
            }
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::FramedRowTooShort { actual } => write!(
                formatter,
                "framed row is shorter than its column-count field: {actual} bytes"
            ),
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::TrailingFrameBytes { remaining } => {
                write!(
                    formatter,
                    "trailing bytes in framed row payload: {remaining}"
                )
            }
            QualifiedCodecPrimary::ConstructionEnvelope(error) => {
                std::fmt::Display::fmt(error, formatter)
            }
            #[cfg(feature = "spill")]
            QualifiedCodecPrimary::RetainedBeyondEnvelope { observed, envelope } => write!(
                formatter,
                "decoded row retains {observed} bytes beyond its {envelope}-byte construction envelope"
            ),
        }
    }
}

// Deliberately no `source()` chain: a hard-qualified caller must not borrow a
// provider/allocator error, downcast an Arc-backed payload, and clone it past
// the authority retained by the enclosing diagnostic owner. `Display` and the
// copy-only `kind()` classification preserve useful diagnostics without a
// detachable reference.
impl std::error::Error for QualifiedCodecError {}

/// Decode adapter that makes a known frame boundary visible to the
/// allocation-bearing value readers without changing the public streaming
/// codec APIs.
struct DecodeReader<'a, R: Read + ?Sized> {
    inner: &'a mut R,
    framed_remaining: Option<usize>,
}

impl<'a, R: Read + ?Sized> DecodeReader<'a, R> {
    fn unframed(inner: &'a mut R) -> Self {
        Self {
            inner,
            framed_remaining: None,
        }
    }

    #[cfg(feature = "spill")]
    fn framed(inner: &'a mut R, framed_bytes: usize) -> Self {
        Self {
            inner,
            framed_remaining: Some(framed_bytes),
        }
    }

    fn ensure_framed_bytes(
        &self,
        len: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        if let Some(remaining) = self.framed_remaining
            && len > remaining
        {
            return Err(QualifiedCodecPrimary::FramedLengthExceeds {
                description,
                length: len,
                remaining,
            }
            .into());
        }
        Ok(())
    }

    fn read_exact_qualified(&mut self, buffer: &mut [u8]) -> Result<(), QualifiedCodecError> {
        if let Some(remaining) = self.framed_remaining {
            if buffer.len() > remaining {
                return Err(QualifiedCodecPrimary::TruncatedFrame {
                    requested: buffer.len(),
                    remaining,
                }
                .into());
            }
            // The only framed constructor is the private exact-slice entry
            // point below. Once the boundary check succeeds, `Read for &[u8]`
            // cannot return a provider-owned error; this `Io` mapping remains
            // solely to keep the shared generic core usable by unframed
            // compatibility readers.
            self.inner
                .read_exact(buffer)
                .map_err(QualifiedCodecPrimary::Io)?;
            self.framed_remaining = Some(remaining - buffer.len());
            return Ok(());
        }
        self.inner
            .read_exact(buffer)
            .map_err(QualifiedCodecPrimary::Io)
            .map_err(Into::into)
    }

    #[cfg(feature = "spill")]
    fn remaining_framed_bytes(&self) -> Result<usize, QualifiedCodecError> {
        self.framed_remaining
            .ok_or(QualifiedCodecPrimary::MissingFrameBoundary)
            .map_err(Into::into)
    }
}

impl<R: Read + ?Sized> Read for DecodeReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.framed_remaining else {
            return self.inner.read(buffer);
        };
        let allowed = remaining.min(buffer.len());
        let read = self.inner.read(&mut buffer[..allowed])?;
        self.framed_remaining = Some(remaining - read);
        Ok(read)
    }
}

#[derive(Clone, Copy, Debug)]
struct CounterSortEntry {
    key_start: usize,
    key_end: usize,
    value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ObservedCounterScratch {
    pub(crate) entry_capacity: usize,
    pub(crate) key_capacity: usize,
}

/// Reusable, owned workspace for deterministic counter-map encoding.
///
/// Keys are copied into an arena and entries retain only checked spans into
/// that arena. This avoids borrowed-lifetime escape hatches while allowing one
/// fallible preparation before an external-sort run creates any files.
#[derive(Default)]
pub(crate) struct CounterSortScratch {
    entries: Vec<CounterSortEntry>,
    key_bytes: Vec<u8>,
    prepared_entries: usize,
    prepared_key_bytes: usize,
    #[cfg(test)]
    write_growths: usize,
}

impl CounterSortScratch {
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            key_bytes: Vec::new(),
            prepared_entries: 0,
            prepared_key_bytes: 0,
            #[cfg(test)]
            write_growths: 0,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        required_entries: usize,
        required_key_bytes: usize,
    ) -> std::io::Result<ObservedCounterScratch> {
        self.clear();
        let _ =
            scratch_capacity_bytes::<CounterSortEntry>(required_entries, "counter sort entries")?;
        let _ = scratch_capacity_bytes::<u8>(required_key_bytes, "counter sort key bytes")?;

        if required_entries > self.entries.capacity() {
            self.entries
                .try_reserve_exact(required_entries)
                .map_err(|error| scratch_allocation_error("counter sort entries", error))?;
        }
        if required_key_bytes > self.key_bytes.capacity() {
            self.key_bytes
                .try_reserve_exact(required_key_bytes)
                .map_err(|error| scratch_allocation_error("counter sort key bytes", error))?;
        }

        self.prepared_entries = required_entries;
        self.prepared_key_bytes = required_key_bytes;
        Ok(self.observed_capacity())
    }

    fn ensure_prepared(
        &self,
        required_entries: usize,
        required_key_bytes: usize,
    ) -> std::io::Result<()> {
        if !self.entries.is_empty() || !self.key_bytes.is_empty() {
            return Err(std::io::Error::other(
                "counter sort scratch was not cleared after its previous use",
            ));
        }
        if required_entries > self.prepared_entries || required_key_bytes > self.prepared_key_bytes
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!(
                    "counter sort workspace ({required_entries} entries, {required_key_bytes} key bytes) exceeds prepared limits ({} entries, {} key bytes)",
                    self.prepared_entries, self.prepared_key_bytes
                ),
            ));
        }
        if required_entries > self.entries.capacity()
            || required_key_bytes > self.key_bytes.capacity()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "prepared counter sort capacity invariant violated",
            ));
        }
        Ok(())
    }

    fn map_scope<'a>(
        &'a mut self,
        map: &std::collections::HashMap<String, u64>,
    ) -> std::io::Result<CounterSortUse<'a>> {
        let mut usage = CounterSortUse { scratch: self };
        usage.load(map)?;
        Ok(usage)
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.key_bytes.clear();
    }

    #[cfg(feature = "spill")]
    pub(crate) fn discard_capacity(&mut self) {
        self.entries = Vec::new();
        self.key_bytes = Vec::new();
        self.prepared_entries = 0;
        self.prepared_key_bytes = 0;
    }

    pub(crate) fn observed_capacity(&self) -> ObservedCounterScratch {
        ObservedCounterScratch {
            entry_capacity: self.entries.capacity(),
            key_capacity: self.key_bytes.capacity(),
        }
    }

    pub(crate) fn requested_capacity_bytes(
        required_entries: usize,
        required_key_bytes: usize,
    ) -> std::io::Result<usize> {
        let entry_bytes =
            scratch_capacity_bytes::<CounterSortEntry>(required_entries, "counter sort entries")?;
        let key_bytes = scratch_capacity_bytes::<u8>(required_key_bytes, "counter sort key bytes")?;
        entry_bytes.checked_add(key_bytes).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "counter sort workspace exceeds the platform address space",
            )
        })
    }

    pub(crate) fn observed_capacity_bytes(&self) -> std::io::Result<usize> {
        Self::requested_capacity_bytes(self.entries.capacity(), self.key_bytes.capacity())
    }

    #[cfg(test)]
    pub(crate) fn entry_len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn key_len(&self) -> usize {
        self.key_bytes.len()
    }

    #[cfg(test)]
    pub(crate) fn entry_capacity(&self) -> usize {
        self.entries.capacity()
    }

    #[cfg(test)]
    pub(crate) fn key_capacity(&self) -> usize {
        self.key_bytes.capacity()
    }

    #[cfg(test)]
    pub(crate) fn entry_pointer(&self) -> *const () {
        self.entries.as_ptr().cast()
    }

    #[cfg(test)]
    pub(crate) fn key_pointer(&self) -> *const u8 {
        self.key_bytes.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn write_growths(&self) -> usize {
        self.write_growths
    }
}

fn scratch_capacity_bytes<T>(required: usize, description: &str) -> std::io::Result<usize> {
    let layout = std::alloc::Layout::array::<T>(required).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            format!("{description} exceed the platform address space"),
        )
    })?;
    if layout.size() > isize::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            format!("{description} exceed the platform address space"),
        ));
    }
    Ok(layout.size())
}

fn scratch_allocation_error(description: &str, error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::OutOfMemory,
        format!("failed to reserve {description}: {error}"),
    )
}

struct CounterSortUse<'a> {
    scratch: &'a mut CounterSortScratch,
}

impl CounterSortUse<'_> {
    fn load(&mut self, map: &std::collections::HashMap<String, u64>) -> std::io::Result<()> {
        let required_key_bytes = map.keys().try_fold(0usize, |total, key| {
            total
                .checked_add(key.len())
                .ok_or_else(|| invalid_input("counter sort key bytes overflow"))
        })?;
        self.scratch
            .ensure_prepared(map.len(), required_key_bytes)?;

        for (key, value) in map {
            let key_start = self.scratch.key_bytes.len();
            let key_end = key_start
                .checked_add(key.len())
                .ok_or_else(|| invalid_input("counter sort key offset overflow"))?;
            if key_end > self.scratch.key_bytes.capacity()
                || self.scratch.entries.len() == self.scratch.entries.capacity()
            {
                #[cfg(test)]
                {
                    self.scratch.write_growths = self.scratch.write_growths.saturating_add(1);
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "prepared counter sort capacity invariant violated",
                ));
            }
            self.scratch.key_bytes.extend_from_slice(key.as_bytes());
            self.scratch.entries.push(CounterSortEntry {
                key_start,
                key_end,
                value: *value,
            });
        }
        if self.scratch.entries.len() != map.len()
            || self.scratch.key_bytes.len() != required_key_bytes
        {
            return Err(invalid_input(
                "counter iterator differed from its measured map workspace",
            ));
        }

        let key_bytes = self.scratch.key_bytes.as_slice();
        self.scratch.entries.sort_unstable_by(|left, right| {
            key_bytes[left.key_start..left.key_end].cmp(&key_bytes[right.key_start..right.key_end])
        });
        Ok(())
    }

    fn write_to<W: Write + ?Sized>(&self, w: &mut W) -> std::io::Result<usize> {
        write_len(self.scratch.entries.len(), w)?;
        let mut total = 8;
        for entry in &self.scratch.entries {
            let key = self
                .scratch
                .key_bytes
                .get(entry.key_start..entry.key_end)
                .ok_or_else(|| invalid_input("counter sort key span is out of bounds"))?;
            write_len(key.len(), w)?;
            w.write_all(key)?;
            add_encoded_size(&mut total, 8)?;
            add_encoded_size(&mut total, key.len())?;
            w.write_all(&entry.value.to_le_bytes())?;
            add_encoded_size(&mut total, 8)?;
        }
        Ok(total)
    }
}

impl Drop for CounterSortUse<'_> {
    fn drop(&mut self) {
        self.scratch.clear();
    }
}

/// Shared custom-state decode budget used across multiple embedded Values.
#[cfg(feature = "spill")]
pub struct SpillCodecDecodeBudget {
    budget: CodecBudget,
}

/// Shared custom-state encode budget across outer slots and embedded Values.
#[cfg(feature = "spill")]
pub struct SpillCodecEncodeBudget {
    budget: CodecBudget,
    counter_scratch: CounterSortScratch,
}

#[cfg(feature = "spill")]
impl SpillCodecEncodeBudget {
    /// Creates one cumulative encode budget for a complete custom-state record.
    #[must_use]
    pub fn new(limits: CodecLimits) -> Self {
        Self {
            budget: CodecBudget::new(limits),
            counter_scratch: CounterSortScratch::new(),
        }
    }

    /// Charges resident storage for `count` outer slots of type `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the count or its resident size exceeds the shared
    /// item or byte limit.
    pub fn charge_items<T>(&mut self, count: usize, description: &str) -> std::io::Result<()> {
        self.budget.charge_items(
            count,
            std::mem::size_of::<T>(),
            std::io::ErrorKind::InvalidInput,
            description,
        )
    }

    /// Charges backing storage bytes without introducing logical collection items.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage exceeds this record's remaining byte grant.
    pub fn charge_storage_bytes(&mut self, bytes: usize, description: &str) -> std::io::Result<()> {
        self.budget
            .charge_bytes(bytes, 1, std::io::ErrorKind::InvalidInput, description)
    }

    /// Encodes one embedded value against this record's remaining budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is unsupported, exceeds the remaining
    /// shared limits, cannot prepare deterministic scratch, or cannot be
    /// written.
    pub fn encode_value<W: Write + ?Sized>(
        &mut self,
        value: &Value,
        writer: &mut W,
    ) -> std::io::Result<usize> {
        validate_value(value, &mut self.budget, 0)?;
        let measurement = measure_value_encoding(value)?;
        self.counter_scratch.prepare(
            measurement.counter_sort_entries,
            measurement.counter_sort_key_bytes,
        )?;
        serialize_value_unchecked(value, writer, &mut self.counter_scratch)
    }
}

#[cfg(feature = "spill")]
impl SpillCodecDecodeBudget {
    /// Creates one cumulative decode budget for a complete custom-state record.
    #[must_use]
    pub fn new(limits: CodecLimits) -> Self {
        Self {
            budget: CodecBudget::new(limits),
        }
    }

    /// Charges resident storage for `count` decoded outer slots of type `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the count or its resident size exceeds the shared
    /// item or byte limit.
    pub fn charge_items<T>(&mut self, count: usize, description: &str) -> std::io::Result<()> {
        self.budget.charge_items(
            count,
            std::mem::size_of::<T>(),
            std::io::ErrorKind::InvalidData,
            description,
        )
    }

    /// Charges backing storage bytes before allocation, without introducing items.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage exceeds this record's remaining byte grant.
    pub fn charge_storage_bytes(&mut self, bytes: usize, description: &str) -> std::io::Result<()> {
        self.budget
            .charge_bytes(bytes, 1, std::io::ErrorKind::InvalidData, description)
    }

    /// Decodes one embedded value against this record's remaining budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is malformed, exceeds the remaining
    /// shared limits, cannot reserve memory, or cannot be read completely.
    pub fn decode_value<R: Read + ?Sized>(&mut self, reader: &mut R) -> std::io::Result<Value> {
        let mut reader = DecodeReader::unframed(reader);
        let mut payload = DecodedPayloadAccumulator::discarding();
        deserialize_value_with_budget(&mut reader, &mut self.budget, &mut payload, 0)
            .map_err(QualifiedCodecError::into_io)
    }
}

impl CodecBudget {
    fn new(limits: CodecLimits) -> Self {
        Self {
            limits,
            remaining_bytes: limits.max_bytes,
            remaining_items: limits.max_items,
        }
    }

    fn charge_bytes(
        &mut self,
        count: usize,
        item_size: usize,
        kind: std::io::ErrorKind,
        description: &str,
    ) -> std::io::Result<()> {
        let bytes = count.checked_mul(item_size).ok_or_else(|| {
            std::io::Error::new(kind, format!("{description} allocation size overflow"))
        })?;
        self.remaining_bytes = self.remaining_bytes.checked_sub(bytes).ok_or_else(|| {
            std::io::Error::new(
                kind,
                format!(
                    "{description} exceeds cumulative {}-byte codec budget",
                    self.limits.max_bytes
                ),
            )
        })?;
        Ok(())
    }

    fn charge_items(
        &mut self,
        count: usize,
        item_size: usize,
        kind: std::io::ErrorKind,
        description: &str,
    ) -> std::io::Result<()> {
        if count > self.limits.max_items {
            return Err(std::io::Error::new(
                kind,
                format!(
                    "{description} count {count} exceeds maximum {} (codec item limit)",
                    self.limits.max_items
                ),
            ));
        }
        self.remaining_items = self.remaining_items.checked_sub(count).ok_or_else(|| {
            std::io::Error::new(
                kind,
                format!(
                    "{description} exceeds cumulative {}-item codec budget",
                    self.limits.max_items
                ),
            )
        })?;
        self.charge_bytes(count, item_size, kind, description)
    }

    fn charge_bytes_qualified(
        &mut self,
        count: usize,
        item_size: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        let bytes =
            count
                .checked_mul(item_size)
                .ok_or(QualifiedCodecPrimary::BudgetSizeOverflow {
                    description,
                    count,
                    item_size,
                })?;
        let remaining = self.remaining_bytes;
        self.remaining_bytes =
            remaining
                .checked_sub(bytes)
                .ok_or(QualifiedCodecPrimary::ByteBudgetExceeded {
                    description,
                    requested: bytes,
                    remaining,
                    maximum: self.limits.max_bytes,
                })?;
        Ok(())
    }

    fn charge_items_qualified(
        &mut self,
        count: usize,
        item_size: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        if count > self.limits.max_items {
            return Err(QualifiedCodecPrimary::ItemLimitExceeded {
                description,
                count,
                maximum: self.limits.max_items,
            }
            .into());
        }
        let remaining = self.remaining_items;
        self.remaining_items =
            remaining
                .checked_sub(count)
                .ok_or(QualifiedCodecPrimary::ItemBudgetExceeded {
                    description,
                    requested: count,
                    remaining,
                    maximum: self.limits.max_items,
                })?;
        self.charge_bytes_qualified(count, item_size, description)
    }
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}

fn check_depth(depth: usize, max_depth: usize, kind: std::io::ErrorKind) -> std::io::Result<()> {
    if depth > max_depth {
        return Err(std::io::Error::new(
            kind,
            format!("value nesting depth exceeds maximum {max_depth}"),
        ));
    }
    Ok(())
}

fn check_depth_qualified(depth: usize, maximum: usize) -> Result<(), QualifiedCodecError> {
    if depth > maximum {
        return Err(QualifiedCodecPrimary::DepthExceeded { depth, maximum }.into());
    }
    Ok(())
}

fn check_len(
    len: usize,
    max: usize,
    kind: std::io::ErrorKind,
    description: &str,
) -> std::io::Result<()> {
    if len > max {
        return Err(std::io::Error::new(
            kind,
            format!("{description} length {len} exceeds maximum {max}"),
        ));
    }
    Ok(())
}

fn validate_string(
    value: &str,
    budget: &mut CodecBudget,
    kind: std::io::ErrorKind,
    description: &str,
) -> std::io::Result<()> {
    check_len(value.len(), budget.limits.max_bytes, kind, description)?;
    budget.charge_bytes(value.len(), 1, kind, description)
}

fn validate_value(value: &Value, budget: &mut CodecBudget, depth: usize) -> std::io::Result<()> {
    #[cfg(test)]
    SEMANTIC_KEY_TRAVERSAL_VISITS.with(|visits| {
        let (validation, measurement) = visits.get();
        visits.set((validation + 1, measurement));
    });
    let kind = std::io::ErrorKind::InvalidInput;
    check_depth(depth, budget.limits.max_depth, kind)?;
    match value {
        Value::Null
        | Value::Bool(_)
        | Value::Int64(_)
        | Value::Float64(_)
        | Value::Timestamp(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_)
        | Value::ZonedDatetime(_) => Ok(()),
        Value::String(value) => validate_string(value, budget, kind, "string"),
        Value::Bytes(value) => {
            check_len(value.len(), budget.limits.max_bytes, kind, "byte string")?;
            budget.charge_bytes(value.len(), 1, kind, "byte string")
        }
        Value::List(items) => {
            budget.charge_items(items.len(), std::mem::size_of::<Value>(), kind, "list item")?;
            for item in items.iter() {
                validate_value(item, budget, depth + 1)?;
            }
            Ok(())
        }
        Value::Map(map) => {
            budget.charge_items(
                map.len(),
                std::mem::size_of::<(grafeo_common::types::PropertyKey, Value)>(),
                kind,
                "map entry",
            )?;
            for (key, value) in map.iter() {
                validate_string(key.as_str(), budget, kind, "map key")?;
                validate_value(value, budget, depth + 1)?;
            }
            Ok(())
        }
        Value::Vector(values) => budget.charge_items(
            values.len(),
            std::mem::size_of::<f32>(),
            kind,
            "vector element",
        ),
        Value::Path { nodes, edges } => {
            budget.charge_items(nodes.len(), std::mem::size_of::<Value>(), kind, "path node")?;
            for node in nodes.iter() {
                validate_value(node, budget, depth + 1)?;
            }
            budget.charge_items(edges.len(), std::mem::size_of::<Value>(), kind, "path edge")?;
            for edge in edges.iter() {
                validate_value(edge, budget, depth + 1)?;
            }
            Ok(())
        }
        Value::GCounter(counts) => validate_counter_map(counts, budget, kind),
        Value::OnCounter { pos, neg } => {
            validate_counter_map(pos, budget, kind)?;
            validate_counter_map(neg, budget, kind)
        }
        Value::RdfLiteral {
            lexical,
            language,
            datatype,
        } => {
            validate_string(lexical, budget, kind, "RDF literal lexical form")?;
            if let Some(language) = language {
                validate_string(language, budget, kind, "RDF literal language")?;
            }
            if let Some(datatype) = datatype {
                validate_string(datatype, budget, kind, "RDF literal datatype")?;
            }
            Ok(())
        }
        _ => Err(invalid_input(
            "unsupported Value variant in spill serialization",
        )),
    }
}

fn validate_counter_map(
    map: &std::collections::HashMap<String, u64>,
    budget: &mut CodecBudget,
    kind: std::io::ErrorKind,
) -> std::io::Result<()> {
    budget.charge_items(
        map.len(),
        std::mem::size_of::<(String, u64)>(),
        kind,
        "counter entry",
    )?;
    for key in map.keys() {
        validate_string(key, budget, kind, "counter key")?;
    }
    Ok(())
}

fn write_len<W: Write + ?Sized>(len: usize, w: &mut W) -> std::io::Result<()> {
    let len = u64::try_from(len).map_err(|_| invalid_input("length does not fit spill format"))?;
    w.write_all(&len.to_le_bytes())
}

fn write_string<W: Write + ?Sized>(value: &str, w: &mut W) -> std::io::Result<usize> {
    write_len(value.len(), w)?;
    w.write_all(value.as_bytes())?;
    8usize
        .checked_add(value.len())
        .ok_or_else(|| invalid_input("encoded string size overflow"))
}

fn add_encoded_size(total: &mut usize, additional: usize) -> std::io::Result<()> {
    *total = total
        .checked_add(additional)
        .ok_or_else(|| invalid_input("encoded value size overflow"))?;
    Ok(())
}

fn decoded_wire_byte_retained_bytes() -> Result<usize, MemoryGrantError> {
    std::mem::size_of::<Value>()
        .checked_add(std::mem::size_of::<Vec<Value>>())
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or(MemoryGrantError::ArithmeticOverflow {
            current_bytes: std::mem::size_of::<Value>(),
            additional_bytes: std::mem::size_of::<Vec<Value>>(),
        })
}

const LENGTH_PREFIX_WIRE_BYTES: usize = std::mem::size_of::<u64>();
const SHARED_CONTROL_WIRE_BYTES: usize = 1;
const OPAQUE_MAP_ENTRY_WIRE_BYTES: usize = LENGTH_PREFIX_WIRE_BYTES - SHARED_CONTROL_WIRE_BYTES;

/// Returns the conservative resident envelope for one decoded row payload.
///
/// `encoded_bytes` must be the exact length produced or accepted by this
/// module's checked row codec. Every recursively decoded item consumes at
/// least one wire byte. For each byte, including structural tags and collection
/// lengths, the envelope therefore funds one complete [`Value`] slot, one
/// possible nested `Vec<Value>` header, and sixteen bytes of allocator slack.
/// This covers the current decoder's top-level row growth, nested `Vec`
/// construction, `Arc`/`ArcStr` headers, and collection storage. Fixed-width
/// scalar rows are deliberately overbounded.
///
/// This is a decode/frontier admission authority, not a final
/// `AccountedDataChunk` payload measurement. A move-consuming column builder
/// must keep exact direct column capacity separate and transfer or measure
/// shared payload authority without applying this amplification permanently.
///
/// # Errors
///
/// Returns [`MemoryGrantError::ArithmeticOverflow`] if the conservative
/// envelope is not representable on this platform.
#[cfg(any(feature = "spill", test))]
pub(crate) fn conservative_decoded_row_retained_bytes(
    encoded_bytes: usize,
) -> Result<usize, MemoryGrantError> {
    let per_wire_byte = decoded_wire_byte_retained_bytes()?;
    encoded_bytes
        .checked_mul(per_wire_byte)
        .ok_or(MemoryGrantError::ArithmeticOverflow {
            current_bytes: encoded_bytes,
            additional_bytes: per_wire_byte,
        })
}

/// Decoder-minted proof of the recursively retained heap payload for one row.
///
/// The byte count excludes the decoded row's top-level `Vec<Value>` backing
/// allocation. It covers only allocations reachable through the row's
/// individual [`Value`] elements. Construction and fields are private so a
/// downstream execution stage can move this proof but cannot forge or clone
/// one from an arbitrary post-decode walk.
#[cfg(feature = "spill")]
#[derive(Debug)]
#[must_use = "the payload receipt must remain coupled to its decoded row"]
pub(crate) struct DecodedPayloadReceipt {
    retained_bytes: usize,
}

/// A framed row kept coupled to the decoder proof for its recursive payload.
///
/// Construction and fields are private, and the wrapper is deliberately not
/// cloneable. The next execution stage must consume the whole wrapper to move
/// both values and their decoder-minted receipt forward together.
#[cfg(feature = "spill")]
#[derive(Debug)]
#[must_use = "the decoded values and payload receipt must be consumed together"]
pub(crate) struct DecodedFramedRow {
    values: Vec<Value>,
    receipt: DecodedPayloadReceipt,
}

#[cfg(feature = "spill")]
impl DecodedFramedRow {
    pub(crate) fn values(&self) -> &[Value] {
        &self.values
    }
    /// Consumes the coupled decode result without cloning either component.
    pub(crate) fn into_parts(self) -> (Vec<Value>, DecodedPayloadReceipt) {
        (self.values, self.receipt)
    }
}

#[cfg(feature = "spill")]
impl DecodedPayloadReceipt {
    /// Returns the conservative retained heap bytes witnessed during decode.
    #[must_use]
    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Removes the decoder's known Bytes contribution after its owner has
    /// physically destroyed a private sort trailer. No general resize exists.
    pub(crate) fn release_sort_bytes_tail(&mut self, bytes: usize) -> Result<(), MemoryGrantError> {
        let contribution = decoded_bytes_payload_retained_bytes(bytes)?;
        self.retained_bytes = self.retained_bytes.checked_sub(contribution).ok_or(
            MemoryGrantError::ArithmeticOverflow {
                current_bytes: self.retained_bytes,
                additional_bytes: contribution,
            },
        )?;
        Ok(())
    }
}

/// The same conservative shared-control contribution used by the decoder.
pub(crate) fn decoded_bytes_payload_retained_bytes(
    bytes: usize,
) -> Result<usize, MemoryGrantError> {
    decoded_wire_byte_retained_bytes()?
        .checked_mul(SHARED_CONTROL_WIRE_BYTES)
        .and_then(|control| control.checked_add(bytes))
        .ok_or(MemoryGrantError::ArithmeticOverflow {
            current_bytes: bytes,
            additional_bytes: SHARED_CONTROL_WIRE_BYTES,
        })
}

/// Checked, decode-local accumulation for a `DecodedPayloadReceipt`.
///
/// The recurrence deliberately partitions wire authority by shape. Nested
/// collection slots are charged by their parent, while a child contributes
/// only its own recursively retained heap. Known payload buffers are charged
/// near-linearly. Opaque standard-library map nodes and hash control storage
/// retain a conservative share of their own count/key/value metadata bytes
/// from the broad construction envelope; they never borrow a child subtree's
/// wire bytes.
struct DecodedPayloadAccumulator {
    retained_bytes: Option<usize>,
}

impl DecodedPayloadAccumulator {
    #[cfg(feature = "spill")]
    const fn tracking() -> Self {
        Self {
            retained_bytes: Some(0),
        }
    }

    const fn discarding() -> Self {
        Self {
            retained_bytes: None,
        }
    }

    fn include_allocation(
        &mut self,
        count: usize,
        item_bytes: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        let Some(retained_bytes) = self.retained_bytes else {
            return Ok(());
        };
        let additional = count
            .checked_mul(item_bytes)
            .ok_or(QualifiedCodecPrimary::PayloadRetainedOverflow { description })?;
        self.retained_bytes = Some(
            retained_bytes
                .checked_add(additional)
                .ok_or(QualifiedCodecPrimary::PayloadRetainedOverflow { description })?,
        );
        Ok(())
    }

    fn include_wire_structure(
        &mut self,
        wire_bytes: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        if self.retained_bytes.is_none() {
            return Ok(());
        }
        let per_wire_byte = decoded_wire_byte_retained_bytes()
            .map_err(QualifiedCodecPrimary::ConstructionEnvelope)?;
        self.include_allocation(wire_bytes, per_wire_byte, description)
    }

    fn include_shared_payload(
        &mut self,
        payload_bytes: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        // One length-prefix byte funds Arc/control/layout slack. For a Value
        // payload, its tag remains available to fund the containing slot. For
        // an untagged map/counter key, the other seven prefix bytes belong to
        // that collection's opaque structural reserve.
        self.include_wire_structure(SHARED_CONTROL_WIRE_BYTES, description)?;
        self.include_allocation(payload_bytes, 1, description)
    }

    fn include_value_slice(
        &mut self,
        values: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        // One collection-length byte funds the Arc/control allocation. Each
        // child's tag funds its exact inline Value slot; the child's recursive
        // receipt therefore excludes that slot.
        self.include_wire_structure(SHARED_CONTROL_WIRE_BYTES, description)?;
        self.include_allocation(values, std::mem::size_of::<Value>(), description)
    }

    fn include_opaque_map(
        &mut self,
        entries: usize,
        description: &'static str,
    ) -> Result<(), QualifiedCodecError> {
        if self.retained_bytes.is_none() {
            return Ok(());
        }
        // All eight count bytes fund the Arc, root, and empty-map structure.
        // On the supported 32/64-bit Rust toolchains, eight complete envelope
        // units cover the worst one-node BTreeMap leaf plus Arc<Map>, and also
        // exceed a one-entry HashMap table plus Arc<HashMap>. Each subsequent
        // entry contributes seven more units: one key-length byte belongs to
        // the retained key allocation, while the remaining seven cover direct
        // key/value storage and any additional opaque BTree node, Hash bucket,
        // or control bytes. Internal BTree edge arrays arise only after enough
        // entries have contributed multiple such units. This keeps the proof
        // on the established wire envelope instead of depending on allocator
        // usable-size or private collection-capacity inspection.
        // This 8+7 reserve is a supported toolchain/target ABI qualification
        // assumption and must be requalified when Rust or target layouts
        // change.
        //
        // Child Values and key payload bytes fund themselves independently.
        let entry_wire_bytes = entries
            .checked_mul(OPAQUE_MAP_ENTRY_WIRE_BYTES)
            .ok_or(QualifiedCodecPrimary::PayloadRetainedOverflow { description })?;
        let structural_wire_bytes = LENGTH_PREFIX_WIRE_BYTES
            .checked_add(entry_wire_bytes)
            .ok_or(QualifiedCodecPrimary::PayloadRetainedOverflow { description })?;
        self.include_wire_structure(structural_wire_bytes, description)
    }

    #[cfg(feature = "spill")]
    fn into_receipt(self) -> Result<DecodedPayloadReceipt, QualifiedCodecError> {
        self.retained_bytes
            .map(|retained_bytes| DecodedPayloadReceipt { retained_bytes })
            .ok_or(QualifiedCodecPrimary::PayloadTrackingDisabled)
            .map_err(Into::into)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SerializedRowMeasurement {
    pub(crate) encoded_bytes: usize,
    pub(crate) counter_sort_entries: usize,
    pub(crate) counter_sort_key_bytes: usize,
}

impl SerializedRowMeasurement {
    const fn empty_row() -> Self {
        Self {
            encoded_bytes: 8,
            counter_sort_entries: 0,
            counter_sort_key_bytes: 0,
        }
    }

    fn include(&mut self, value: Self) -> std::io::Result<()> {
        add_encoded_size(&mut self.encoded_bytes, value.encoded_bytes)?;
        self.counter_sort_entries = self.counter_sort_entries.max(value.counter_sort_entries);
        self.counter_sort_key_bytes = self
            .counter_sort_key_bytes
            .max(value.counter_sort_key_bytes);
        Ok(())
    }
}

fn measured_bytes(encoded_bytes: usize) -> SerializedRowMeasurement {
    SerializedRowMeasurement {
        encoded_bytes,
        counter_sort_entries: 0,
        counter_sort_key_bytes: 0,
    }
}

fn measure_string_encoding(value: &str) -> std::io::Result<usize> {
    8usize
        .checked_add(value.len())
        .ok_or_else(|| invalid_input("encoded string size overflow"))
}

fn measure_counter_map_encoding(
    map: &std::collections::HashMap<String, u64>,
) -> std::io::Result<SerializedRowMeasurement> {
    let mut measurement = SerializedRowMeasurement {
        encoded_bytes: 8,
        counter_sort_entries: map.len(),
        counter_sort_key_bytes: 0,
    };
    for key in map.keys() {
        add_encoded_size(&mut measurement.counter_sort_key_bytes, key.len())?;
        add_encoded_size(
            &mut measurement.encoded_bytes,
            measure_string_encoding(key)?,
        )?;
        add_encoded_size(&mut measurement.encoded_bytes, 8)?;
    }
    Ok(measurement)
}

fn measure_value_encoding(value: &Value) -> std::io::Result<SerializedRowMeasurement> {
    #[cfg(test)]
    SEMANTIC_KEY_TRAVERSAL_VISITS.with(|visits| {
        let (validation, measurement) = visits.get();
        visits.set((validation, measurement + 1));
    });
    match value {
        Value::Null => Ok(measured_bytes(1)),
        Value::Bool(_) => Ok(measured_bytes(2)),
        Value::Int64(_) | Value::Float64(_) | Value::Timestamp(_) => Ok(measured_bytes(9)),
        Value::String(value) => {
            let mut measurement = measured_bytes(1);
            add_encoded_size(
                &mut measurement.encoded_bytes,
                measure_string_encoding(value)?,
            )?;
            Ok(measurement)
        }
        Value::Bytes(value) => {
            let mut measurement = measured_bytes(9);
            add_encoded_size(&mut measurement.encoded_bytes, value.len())?;
            Ok(measurement)
        }
        Value::List(items) => {
            let mut measurement = measured_bytes(9);
            for item in items.iter() {
                measurement.include(measure_value_encoding(item)?)?;
            }
            Ok(measurement)
        }
        Value::Map(map) => {
            let mut measurement = measured_bytes(9);
            for (key, value) in map.iter() {
                add_encoded_size(
                    &mut measurement.encoded_bytes,
                    measure_string_encoding(key.as_str())?,
                )?;
                measurement.include(measure_value_encoding(value)?)?;
            }
            Ok(measurement)
        }
        Value::Vector(values) => {
            let payload = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| invalid_input("encoded vector size overflow"))?;
            let mut measurement = measured_bytes(9);
            add_encoded_size(&mut measurement.encoded_bytes, payload)?;
            Ok(measurement)
        }
        Value::Date(_) => Ok(measured_bytes(5)),
        Value::Time(value) => Ok(measured_bytes(if value.offset_seconds().is_some() {
            14
        } else {
            10
        })),
        Value::Duration(_) => Ok(measured_bytes(25)),
        Value::ZonedDatetime(_) => Ok(measured_bytes(13)),
        Value::Path { nodes, edges } => {
            let mut measurement = measured_bytes(17);
            for node in nodes.iter() {
                measurement.include(measure_value_encoding(node)?)?;
            }
            for edge in edges.iter() {
                measurement.include(measure_value_encoding(edge)?)?;
            }
            Ok(measurement)
        }
        Value::GCounter(counts) => {
            let mut measurement = measured_bytes(1);
            measurement.include(measure_counter_map_encoding(counts)?)?;
            Ok(measurement)
        }
        Value::OnCounter { pos, neg } => {
            let mut measurement = measured_bytes(1);
            measurement.include(measure_counter_map_encoding(pos)?)?;
            measurement.include(measure_counter_map_encoding(neg)?)?;
            Ok(measurement)
        }
        Value::RdfLiteral {
            lexical,
            language,
            datatype,
        } => {
            let mut measurement = measured_bytes(1);
            add_encoded_size(
                &mut measurement.encoded_bytes,
                measure_string_encoding(lexical)?,
            )?;
            for optional in [language.as_deref(), datatype.as_deref()] {
                add_encoded_size(&mut measurement.encoded_bytes, 1)?;
                if let Some(value) = optional {
                    add_encoded_size(
                        &mut measurement.encoded_bytes,
                        measure_string_encoding(value)?,
                    )?;
                }
            }
            Ok(measurement)
        }
        _ => Err(invalid_input(
            "unsupported Value variant in spill serialization",
        )),
    }
}

fn read_bounded_len<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    max: usize,
    description: &'static str,
) -> Result<usize, QualifiedCodecError> {
    let mut len_buf = [0u8; 8];
    r.read_exact_qualified(&mut len_buf)?;
    let encoded = u64::from_le_bytes(len_buf);
    let len =
        usize::try_from(encoded).map_err(|_| QualifiedCodecPrimary::LengthNotAddressable {
            description,
            encoded,
        })?;
    if len > max {
        return Err(QualifiedCodecPrimary::LengthExceeds {
            description,
            length: len,
            maximum: max,
        }
        .into());
    }
    Ok(len)
}

fn read_string<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    budget: &mut CodecBudget,
    description: &'static str,
) -> Result<ArcStr, QualifiedCodecError> {
    let bytes = read_byte_buffer(r, budget, description)?;
    let value = std::str::from_utf8(&bytes)
        .map_err(|error| QualifiedCodecPrimary::InvalidUtf8 { description, error })?;
    if value.is_empty() {
        return Ok(ArcStr::new());
    }
    // `try_alloc` copies exactly the validated string slice into the final
    // ArcStr allocation. The decoder does not transfer spare capacity from its
    // temporary byte buffer, so `value.len()` is the retained payload extent.
    ArcStr::try_alloc(value)
        .ok_or(QualifiedCodecPrimary::SharedAllocation { description })
        .map_err(Into::into)
}

fn read_byte_buffer<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    budget: &mut CodecBudget,
    description: &'static str,
) -> Result<Vec<u8>, QualifiedCodecError> {
    let len = read_bounded_len(r, budget.limits.max_bytes, description)?;
    // The declared resident length can fit its grant while being impossible
    // for this specific frame. Refuse before the first allocation attempt.
    r.ensure_framed_bytes(len, description)?;
    budget.charge_bytes_qualified(len, 1, description)?;
    read_bytes_incrementally(r, len, description)
}

fn read_bytes_incrementally<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    len: usize,
    description: &'static str,
) -> Result<Vec<u8>, QualifiedCodecError> {
    let mut bytes = Vec::new();
    while bytes.len() < len {
        let chunk_len = (len - bytes.len()).min(DECODE_GROWTH_CHUNK);
        bytes
            .try_reserve_exact(chunk_len)
            .map_err(|error| QualifiedCodecPrimary::Allocation { description, error })?;
        let start = bytes.len();
        bytes.resize(start + chunk_len, 0);
        r.read_exact_qualified(&mut bytes[start..])?;
    }
    Ok(bytes)
}

fn try_push<T>(
    values: &mut Vec<T>,
    value: T,
    description: &'static str,
) -> Result<(), QualifiedCodecError> {
    if values.len() == values.capacity() {
        values
            .try_reserve(1)
            .map_err(|error| QualifiedCodecPrimary::Allocation { description, error })?;
    }
    values.push(value);
    Ok(())
}

fn write_counter_map<W: Write + ?Sized>(
    map: &std::collections::HashMap<String, u64>,
    w: &mut W,
    scratch: &mut CounterSortScratch,
) -> std::io::Result<usize> {
    scratch.map_scope(map)?.write_to(w)
}

/// Serializes a Value to bytes.
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn serialize_value<W: Write + ?Sized>(value: &Value, w: &mut W) -> std::io::Result<usize> {
    serialize_value_with_limits(value, w, CodecLimits::format_max())
}

/// Serializes one value under caller-supplied safety/resource limits.
///
/// # Errors
///
/// Returns an error before writing if the value exceeds `limits`, or if the
/// writer fails.
pub fn serialize_value_with_limits<W: Write + ?Sized>(
    value: &Value,
    w: &mut W,
    limits: CodecLimits,
) -> std::io::Result<usize> {
    let mut budget = CodecBudget::new(limits);
    validate_value(value, &mut budget, 0)?;
    let measurement = measure_value_encoding(value)?;
    let mut scratch = CounterSortScratch::new();
    scratch.prepare(
        measurement.counter_sort_entries,
        measurement.counter_sort_key_bytes,
    )?;
    serialize_value_unchecked(value, w, &mut scratch)
}

fn serialize_value_unchecked<W: Write + ?Sized>(
    value: &Value,
    w: &mut W,
    counter_scratch: &mut CounterSortScratch,
) -> std::io::Result<usize> {
    serialize_value_mode::<false, W>(value, w, counter_scratch)
}

fn serialize_value_mode<const SEMANTIC_KEY: bool, W: Write + ?Sized>(
    value: &Value,
    w: &mut W,
    counter_scratch: &mut CounterSortScratch,
) -> std::io::Result<usize> {
    match value {
        Value::Null => {
            w.write_all(&[TAG_NULL])?;
            Ok(1)
        }
        Value::Bool(b) => {
            w.write_all(&[TAG_BOOL, u8::from(*b)])?;
            Ok(2)
        }
        Value::Int64(i) => {
            w.write_all(&[TAG_INT64])?;
            w.write_all(&i.to_le_bytes())?;
            Ok(9)
        }
        Value::Float64(f) => {
            w.write_all(&[TAG_FLOAT64])?;
            let bits = if SEMANTIC_KEY {
                grafeo_common::types::canonical_f64_bits(*f)
            } else {
                f.to_bits()
            };
            w.write_all(&bits.to_le_bytes())?;
            Ok(9)
        }
        Value::String(s) => {
            w.write_all(&[TAG_STRING])?;
            let mut total = 1;
            add_encoded_size(&mut total, write_string(s, w)?)?;
            Ok(total)
        }
        Value::Bytes(b) => {
            w.write_all(&[TAG_BYTES])?;
            write_len(b.len(), w)?;
            w.write_all(b)?;
            let mut total = 9;
            add_encoded_size(&mut total, b.len())?;
            Ok(total)
        }
        Value::Timestamp(t) => {
            w.write_all(&[TAG_TIMESTAMP])?;
            // Timestamp is internally an i64 (microseconds since epoch)
            let micros = t.as_micros();
            w.write_all(&micros.to_le_bytes())?;
            Ok(9)
        }
        Value::List(items) => {
            w.write_all(&[TAG_LIST])?;
            write_len(items.len(), w)?;
            let mut total = 1 + 8;
            for item in items.iter() {
                add_encoded_size(
                    &mut total,
                    serialize_value_mode::<SEMANTIC_KEY, W>(item, w, counter_scratch)?,
                )?;
            }
            Ok(total)
        }
        Value::Map(map) => {
            w.write_all(&[TAG_MAP])?;
            write_len(map.len(), w)?;
            let mut total = 1 + 8;
            for (key, val) in map.iter() {
                // Serialize key as string
                add_encoded_size(&mut total, write_string(key.as_str(), w)?)?;
                // Serialize value
                add_encoded_size(
                    &mut total,
                    serialize_value_mode::<SEMANTIC_KEY, W>(val, w, counter_scratch)?,
                )?;
            }
            Ok(total)
        }
        Value::Vector(v) => {
            w.write_all(&[TAG_VECTOR])?;
            write_len(v.len(), w)?;
            for &f in v.iter() {
                w.write_all(&f.to_le_bytes())?;
            }
            let payload = v
                .len()
                .checked_mul(4)
                .ok_or_else(|| invalid_input("encoded vector size overflow"))?;
            let mut total = 9;
            add_encoded_size(&mut total, payload)?;
            Ok(total)
        }
        Value::Date(d) => {
            w.write_all(&[TAG_DATE])?;
            w.write_all(&d.as_days().to_le_bytes())?;
            Ok(5)
        }
        Value::Time(t) => {
            w.write_all(&[TAG_TIME_EXACT])?;
            w.write_all(&t.as_nanos().to_le_bytes())?;
            match t.offset_seconds() {
                None => {
                    w.write_all(&[0])?;
                    Ok(10)
                }
                Some(offset) => {
                    w.write_all(&[1])?;
                    w.write_all(&offset.to_le_bytes())?;
                    Ok(14)
                }
            }
        }
        Value::Duration(d) => {
            w.write_all(&[TAG_DURATION])?;
            w.write_all(&d.months().to_le_bytes())?;
            w.write_all(&d.days().to_le_bytes())?;
            w.write_all(&d.nanos().to_le_bytes())?;
            Ok(25)
        }
        Value::ZonedDatetime(zdt) => {
            w.write_all(&[TAG_ZONED_DATETIME])?;
            w.write_all(&zdt.as_timestamp().as_micros().to_le_bytes())?;
            // Native equality compares zoned datetimes by UTC instant. Only
            // key encoding normalizes the offset; witnesses remain lossless.
            let offset = if SEMANTIC_KEY {
                0
            } else {
                zdt.offset_seconds()
            };
            w.write_all(&offset.to_le_bytes())?;
            Ok(13)
        }
        Value::Path { nodes, edges } => {
            w.write_all(&[TAG_PATH])?;
            write_len(nodes.len(), w)?;
            let mut total = 1 + 8;
            for node in nodes.iter() {
                add_encoded_size(
                    &mut total,
                    serialize_value_mode::<SEMANTIC_KEY, W>(node, w, counter_scratch)?,
                )?;
            }
            write_len(edges.len(), w)?;
            add_encoded_size(&mut total, 8)?;
            for edge in edges.iter() {
                add_encoded_size(
                    &mut total,
                    serialize_value_mode::<SEMANTIC_KEY, W>(edge, w, counter_scratch)?,
                )?;
            }
            Ok(total)
        }
        Value::GCounter(counts) => {
            w.write_all(&[TAG_GCOUNTER])?;
            let mut total = 1;
            add_encoded_size(&mut total, write_counter_map(counts, w, counter_scratch)?)?;
            Ok(total)
        }
        Value::OnCounter { pos, neg } => {
            w.write_all(&[TAG_PNCOUNTER])?;
            let mut total = 1;
            add_encoded_size(&mut total, write_counter_map(pos, w, counter_scratch)?)?;
            add_encoded_size(&mut total, write_counter_map(neg, w, counter_scratch)?)?;
            Ok(total)
        }
        Value::RdfLiteral {
            lexical,
            language,
            datatype,
        } => {
            w.write_all(&[TAG_RDF_LITERAL])?;
            let mut total = 1;
            add_encoded_size(&mut total, write_string(lexical, w)?)?;
            add_encoded_size(&mut total, write_optional_string(language.as_deref(), w)?)?;
            add_encoded_size(&mut total, write_optional_string(datatype.as_deref(), w)?)?;
            Ok(total)
        }
        _ => Err(invalid_input(
            "unsupported Value variant in spill serialization",
        )),
    }
}

fn write_optional_string<W: Write + ?Sized>(
    value: Option<&str>,
    w: &mut W,
) -> std::io::Result<usize> {
    match value {
        None => {
            w.write_all(&[0])?;
            Ok(1)
        }
        Some(value) => {
            w.write_all(&[1])?;
            let mut total = 1;
            add_encoded_size(&mut total, write_string(value, w)?)?;
            Ok(total)
        }
    }
}

/// Deserializes a Value from bytes.
///
/// # Errors
///
/// Returns an error if reading fails or the format is invalid.
pub fn deserialize_value<R: Read + ?Sized>(r: &mut R) -> std::io::Result<Value> {
    deserialize_value_with_limits(r, CodecLimits::format_max())
}

/// Deserializes one value under caller-supplied safety/resource limits.
///
/// # Errors
///
/// Returns an error if input is malformed, exceeds `limits`, cannot reserve
/// memory, or cannot be read completely.
pub fn deserialize_value_with_limits<R: Read + ?Sized>(
    r: &mut R,
    limits: CodecLimits,
) -> std::io::Result<Value> {
    let mut budget = CodecBudget::new(limits);
    let mut payload = DecodedPayloadAccumulator::discarding();
    let mut reader = DecodeReader::unframed(r);
    deserialize_value_with_budget(&mut reader, &mut budget, &mut payload, 0)
        .map_err(QualifiedCodecError::into_io)
}

fn deserialize_value_with_budget<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    budget: &mut CodecBudget,
    payload: &mut DecodedPayloadAccumulator,
    depth: usize,
) -> Result<Value, QualifiedCodecError> {
    check_depth_qualified(depth, budget.limits.max_depth)?;
    let mut tag = [0u8; 1];
    r.read_exact_qualified(&mut tag)?;

    match tag[0] {
        TAG_NULL => Ok(Value::Null),
        TAG_BOOL => {
            let mut buf = [0u8; 1];
            r.read_exact_qualified(&mut buf)?;
            match buf[0] {
                0 => Ok(Value::Bool(false)),
                1 => Ok(Value::Bool(true)),
                value => Err(QualifiedCodecPrimary::InvalidBoolean(value).into()),
            }
        }
        TAG_INT64 => {
            let mut buf = [0u8; 8];
            r.read_exact_qualified(&mut buf)?;
            Ok(Value::Int64(i64::from_le_bytes(buf)))
        }
        TAG_FLOAT64 => {
            let mut buf = [0u8; 8];
            r.read_exact_qualified(&mut buf)?;
            Ok(Value::Float64(f64::from_le_bytes(buf)))
        }
        TAG_STRING => {
            let value = read_string(r, budget, "string")?;
            payload.include_shared_payload(value.len(), "string")?;
            Ok(Value::String(value))
        }
        TAG_BYTES => {
            let bytes_buf = read_byte_buffer(r, budget, "byte string")?;
            payload.include_shared_payload(bytes_buf.len(), "byte string")?;
            Ok(Value::Bytes(Arc::from(bytes_buf)))
        }
        TAG_TIMESTAMP => {
            let mut buf = [0u8; 8];
            r.read_exact_qualified(&mut buf)?;
            let micros = i64::from_le_bytes(buf);
            Ok(Value::Timestamp(
                grafeo_common::types::Timestamp::from_micros(micros),
            ))
        }
        TAG_LIST => {
            let len = read_bounded_len(r, budget.limits.max_items, "list item count")?;
            budget.charge_items_qualified(len, std::mem::size_of::<Value>(), "list item")?;
            payload.include_value_slice(len, "list items")?;
            let mut items = Vec::new();
            for _ in 0..len {
                let item = deserialize_value_with_budget(r, budget, payload, depth + 1)?;
                try_push(&mut items, item, "list items")?;
            }
            Ok(Value::List(Arc::from(items)))
        }
        TAG_MAP => {
            let len = read_bounded_len(r, budget.limits.max_items, "map entry count")?;
            budget.charge_items_qualified(
                len,
                std::mem::size_of::<(grafeo_common::types::PropertyKey, Value)>(),
                "map entry",
            )?;
            payload.include_opaque_map(len, "map structure")?;
            let mut map = BTreeMap::new();
            let mut previous_key: Option<ArcStr> = None;
            for _ in 0..len {
                // Read key
                let key = read_string(r, budget, "map key")?;
                payload.include_shared_payload(key.len(), "map key")?;
                if previous_key
                    .as_ref()
                    .is_some_and(|previous| previous.as_str() >= key.as_str())
                {
                    return Err(QualifiedCodecPrimary::MapKeysNotIncreasing.into());
                }
                // Read value
                let val = deserialize_value_with_budget(r, budget, payload, depth + 1)?;
                map.insert(grafeo_common::types::PropertyKey::new(key.clone()), val);
                previous_key = Some(key);
            }
            Ok(Value::Map(Arc::new(map)))
        }
        TAG_VECTOR => {
            let len = read_bounded_len(r, budget.limits.max_items, "vector element count")?;
            budget.charge_items_qualified(len, std::mem::size_of::<f32>(), "vector element")?;
            payload.include_wire_structure(SHARED_CONTROL_WIRE_BYTES, "vector control")?;
            payload.include_allocation(len, std::mem::size_of::<f32>(), "vector elements")?;
            let mut floats = Vec::new();
            let mut buf = [0u8; 4];
            for _ in 0..len {
                r.read_exact_qualified(&mut buf)?;
                try_push(&mut floats, f32::from_le_bytes(buf), "vector elements")?;
            }
            Ok(Value::Vector(Arc::from(floats)))
        }
        TAG_DATE => {
            let mut buf = [0u8; 4];
            r.read_exact_qualified(&mut buf)?;
            Ok(Value::Date(grafeo_common::types::Date::from_days(
                i32::from_le_bytes(buf),
            )))
        }
        TAG_TIME => {
            let mut nanos_buf = [0u8; 8];
            r.read_exact_qualified(&mut nanos_buf)?;
            let nanos = u64::from_le_bytes(nanos_buf);
            let mut offset_buf = [0u8; 4];
            r.read_exact_qualified(&mut offset_buf)?;
            let offset = i32::from_le_bytes(offset_buf);
            let time = grafeo_common::types::Time::from_nanos(nanos)
                .ok_or(QualifiedCodecPrimary::InvalidTimeNanos(nanos))?;
            if offset == i32::MIN {
                Ok(Value::Time(time))
            } else {
                Ok(Value::Time(time.with_offset(offset)))
            }
        }
        TAG_TIME_EXACT => {
            let mut nanos_buf = [0u8; 8];
            r.read_exact_qualified(&mut nanos_buf)?;
            let nanos = u64::from_le_bytes(nanos_buf);
            let time = grafeo_common::types::Time::from_nanos(nanos)
                .ok_or(QualifiedCodecPrimary::InvalidTimeNanos(nanos))?;
            let mut present = [0u8; 1];
            r.read_exact_qualified(&mut present)?;
            match present[0] {
                0 => Ok(Value::Time(time)),
                1 => {
                    let mut offset_buf = [0u8; 4];
                    r.read_exact_qualified(&mut offset_buf)?;
                    Ok(Value::Time(
                        time.with_offset(i32::from_le_bytes(offset_buf)),
                    ))
                }
                value => Err(QualifiedCodecPrimary::InvalidTimeOffsetPresence(value).into()),
            }
        }
        TAG_DURATION => {
            let mut buf = [0u8; 8];
            r.read_exact_qualified(&mut buf)?;
            let months = i64::from_le_bytes(buf);
            r.read_exact_qualified(&mut buf)?;
            let days = i64::from_le_bytes(buf);
            r.read_exact_qualified(&mut buf)?;
            let nanos = i64::from_le_bytes(buf);
            Ok(Value::Duration(grafeo_common::types::Duration::new(
                months, days, nanos,
            )))
        }
        TAG_ZONED_DATETIME => {
            let mut micros_buf = [0u8; 8];
            r.read_exact_qualified(&mut micros_buf)?;
            let micros = i64::from_le_bytes(micros_buf);
            let mut offset_buf = [0u8; 4];
            r.read_exact_qualified(&mut offset_buf)?;
            let offset = i32::from_le_bytes(offset_buf);
            Ok(Value::ZonedDatetime(
                grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                    grafeo_common::types::Timestamp::from_micros(micros),
                    offset,
                ),
            ))
        }
        TAG_PATH => {
            let nodes_len = read_bounded_len(r, budget.limits.max_items, "path node count")?;
            budget.charge_items_qualified(nodes_len, std::mem::size_of::<Value>(), "path node")?;
            payload.include_value_slice(nodes_len, "path nodes")?;
            let mut nodes = Vec::new();
            for _ in 0..nodes_len {
                let node = deserialize_value_with_budget(r, budget, payload, depth + 1)?;
                try_push(&mut nodes, node, "path nodes")?;
            }
            let edges_len = read_bounded_len(r, budget.limits.max_items, "path edge count")?;
            budget.charge_items_qualified(edges_len, std::mem::size_of::<Value>(), "path edge")?;
            payload.include_value_slice(edges_len, "path edges")?;
            let mut edges = Vec::new();
            for _ in 0..edges_len {
                let edge = deserialize_value_with_budget(r, budget, payload, depth + 1)?;
                try_push(&mut edges, edge, "path edges")?;
            }
            Ok(Value::Path {
                nodes: Arc::from(nodes),
                edges: Arc::from(edges),
            })
        }
        TAG_GCOUNTER => {
            let map = read_counter_map(r, budget, payload, "GCounter")?;
            Ok(Value::GCounter(Arc::new(map)))
        }
        TAG_PNCOUNTER => {
            let pos = read_counter_map(r, budget, payload, "PNCounter positive")?;
            let neg = read_counter_map(r, budget, payload, "PNCounter negative")?;
            Ok(Value::OnCounter {
                pos: Arc::new(pos),
                neg: Arc::new(neg),
            })
        }
        TAG_RDF_LITERAL => {
            let lexical = read_string(r, budget, "RDF literal lexical form")?;
            let language = read_optional_string(r, budget, "RDF literal language")?;
            let datatype = read_optional_string(r, budget, "RDF literal datatype")?;
            payload.include_shared_payload(lexical.len(), "RDF literal lexical form")?;
            if let Some(language) = &language {
                payload.include_shared_payload(language.len(), "RDF literal language")?;
            }
            if let Some(datatype) = &datatype {
                payload.include_shared_payload(datatype.len(), "RDF literal datatype")?;
            }
            Ok(Value::RdfLiteral {
                lexical,
                language,
                datatype,
            })
        }
        value => Err(QualifiedCodecPrimary::UnknownValueTag(value).into()),
    }
}

fn read_optional_string<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    budget: &mut CodecBudget,
    description: &'static str,
) -> Result<Option<ArcStr>, QualifiedCodecError> {
    let mut present = [0u8; 1];
    r.read_exact_qualified(&mut present)?;
    match present[0] {
        0 => Ok(None),
        1 => Ok(Some(read_string(r, budget, description)?)),
        value => {
            Err(QualifiedCodecPrimary::InvalidOptionalStringPresence { description, value }.into())
        }
    }
}

fn read_counter_map<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    budget: &mut CodecBudget,
    payload: &mut DecodedPayloadAccumulator,
    description: &'static str,
) -> Result<std::collections::HashMap<String, u64>, QualifiedCodecError> {
    let count = read_bounded_len(r, budget.limits.max_items, "counter entry count")?;
    budget.charge_items_qualified(count, std::mem::size_of::<(String, u64)>(), "counter entry")?;
    payload.include_opaque_map(count, description)?;
    let mut map = std::collections::HashMap::new();
    let mut u64_buf = [0u8; 8];
    for _ in 0..count {
        let key = read_string(r, budget, "counter key")?;
        let key = key.to_string();
        payload.include_shared_payload(key.capacity(), "counter key")?;
        r.read_exact_qualified(&mut u64_buf)?;
        let value = u64::from_le_bytes(u64_buf);
        map.try_reserve(1)
            .map_err(|error| QualifiedCodecPrimary::Allocation {
                description: "counter entries",
                error,
            })?;
        if map.insert(key, value).is_some() {
            return Err(QualifiedCodecPrimary::DuplicateCounterKey.into());
        }
    }
    Ok(map)
}

/// Serializes a row (slice of Values) to bytes.
///
/// Format: `[num_columns: u64][value1][value2]...`
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn serialize_row<W: Write + ?Sized>(row: &[Value], w: &mut W) -> std::io::Result<usize> {
    serialize_row_with_limits(row, w, CodecLimits::format_max())
}

/// Measures the exact encoded row bytes and deterministic-counter scratch
/// entries/key bytes without allocating or writing.
///
/// The same structural and resource limits as serialization are applied. A
/// successful result can therefore be used to reserve the complete plaintext
/// row payload before encoding begins. Framing, sealing, and file buffering are
/// separate workspace concerns. `counter_sort_entries` is the largest
/// individual counter map encountered recursively; deterministic counter
/// encoding never needs more simultaneously-live entry references than this
/// value. `counter_sort_key_bytes` is the corresponding largest sum of UTF-8
/// key bytes for one counter map; the two componentwise maxima can come from
/// different maps because maps are encoded sequentially.
pub(crate) fn measure_serialized_row_with_limits(
    row: &[Value],
    limits: CodecLimits,
) -> std::io::Result<SerializedRowMeasurement> {
    check_len(
        row.len(),
        limits.max_row_columns,
        std::io::ErrorKind::InvalidInput,
        "row column count",
    )?;
    let mut budget = CodecBudget::new(limits);
    budget.charge_items(
        row.len(),
        std::mem::size_of::<Value>(),
        std::io::ErrorKind::InvalidInput,
        "row column",
    )?;
    for value in row {
        validate_value(value, &mut budget, 0)?;
    }

    let mut measurement = SerializedRowMeasurement::empty_row();
    for value in row {
        measurement.include(measure_value_encoding(value)?)?;
    }
    Ok(measurement)
}

/// Serializes a row under caller-supplied safety/resource limits.
///
/// The complete row is validated before its header is written, so a limit,
/// depth, or unsupported-value refusal never leaves a semantic row prefix.
///
/// # Errors
///
/// Returns an error before writing if the row exceeds `limits`, or if the
/// writer fails.
pub fn serialize_row_with_limits<W: Write + ?Sized>(
    row: &[Value],
    w: &mut W,
    limits: CodecLimits,
) -> std::io::Result<usize> {
    let measurement = measure_serialized_row_with_limits(row, limits)?;
    let mut counter_scratch = CounterSortScratch::new();
    counter_scratch.prepare(
        measurement.counter_sort_entries,
        measurement.counter_sort_key_bytes,
    )?;
    serialize_measured_row(row, w, measurement, &mut counter_scratch)
}

/// Serializes a row using counter workspace prepared by a run-wide preflight.
///
/// The row is revalidated and remeasured before any output, so stale or
/// insufficient preparation fails without leaving a row prefix.
#[cfg(any(test, feature = "spill"))]
pub(crate) fn serialize_row_with_prepared_scratch<W: Write + ?Sized>(
    row: &[Value],
    w: &mut W,
    limits: CodecLimits,
    counter_scratch: &mut CounterSortScratch,
) -> std::io::Result<usize> {
    let measurement = measure_serialized_row_with_limits(row, limits)?;
    serialize_measured_row(row, w, measurement, counter_scratch)
}

/// A validated DISTINCT key measurement tied to the immutable row it describes.
///
/// Float64 signed zeros and zoned-datetime offsets are normalized exactly as
/// `HashableValue` equality requires, including within lists, maps and paths.
/// Variant tags, NaN payloads and vector bits remain discriminated. Private
/// fields prevent replacing the row or measurement between admission and
/// encoding; the shared borrow keeps the row and its nested values immutable.
/// Normalization changes no encoded length or capacity.
/// Lossless witness encoding continues through the ordinary row serializer.
pub(crate) struct MeasuredSemanticKey<'a> {
    row: &'a [Value],
    measurement: SerializedRowMeasurement,
}

impl<'a> MeasuredSemanticKey<'a> {
    pub(crate) fn new(row: &'a [Value], limits: CodecLimits) -> std::io::Result<Self> {
        let measurement = measure_serialized_row_with_limits(row, limits)?;
        Ok(Self { row, measurement })
    }

    /// Supplies the sizes needed to grant and prepare workspace before encoding.
    pub(crate) fn measurement(&self) -> SerializedRowMeasurement {
        self.measurement
    }

    pub(crate) fn serialize_with_prepared_scratch<W: Write + ?Sized>(
        self,
        w: &mut W,
        counter_scratch: &mut CounterSortScratch,
    ) -> std::io::Result<usize> {
        serialize_measured_row_mode::<true, W>(self.row, w, self.measurement, counter_scratch)
    }
}

fn serialize_measured_row<W: Write + ?Sized>(
    row: &[Value],
    w: &mut W,
    measurement: SerializedRowMeasurement,
    counter_scratch: &mut CounterSortScratch,
) -> std::io::Result<usize> {
    serialize_measured_row_mode::<false, W>(row, w, measurement, counter_scratch)
}

fn serialize_measured_row_mode<const SEMANTIC_KEY: bool, W: Write + ?Sized>(
    row: &[Value],
    w: &mut W,
    measurement: SerializedRowMeasurement,
    counter_scratch: &mut CounterSortScratch,
) -> std::io::Result<usize> {
    counter_scratch.ensure_prepared(
        measurement.counter_sort_entries,
        measurement.counter_sort_key_bytes,
    )?;
    write_len(row.len(), w)?;
    let mut total = 8;
    for value in row {
        add_encoded_size(
            &mut total,
            serialize_value_mode::<SEMANTIC_KEY, W>(value, w, counter_scratch)?,
        )?;
    }
    if total != measurement.encoded_bytes {
        return Err(invalid_input(format!(
            "encoded row length {total} differs from measured length {}",
            measurement.encoded_bytes
        )));
    }
    Ok(total)
}

/// Deserializes a row from bytes.
///
/// # Arguments
///
/// * `r` - Reader to read from
/// * `expected_columns` - Expected number of columns (for validation, 0 to skip)
///
/// # Errors
///
/// Returns an error if reading fails or column count mismatches.
pub fn deserialize_row<R: Read + ?Sized>(
    r: &mut R,
    expected_columns: usize,
) -> std::io::Result<Vec<Value>> {
    deserialize_row_with_limits(r, expected_columns, CodecLimits::format_max())
}

/// Deserializes a row under caller-supplied safety/resource limits.
///
/// # Errors
///
/// Returns an error if input is malformed, exceeds `limits`, cannot reserve
/// memory, or has a different nonzero expected column count.
pub fn deserialize_row_with_limits<R: Read + ?Sized>(
    r: &mut R,
    expected_columns: usize,
    limits: CodecLimits,
) -> std::io::Result<Vec<Value>> {
    let mut reader = DecodeReader::unframed(r);
    let mut payload = DecodedPayloadAccumulator::discarding();
    deserialize_row_from_reader(&mut reader, expected_columns, limits, &mut payload)
        .map_err(QualifiedCodecError::into_io)
}

fn deserialize_row_from_reader<R: Read + ?Sized>(
    r: &mut DecodeReader<'_, R>,
    expected_columns: usize,
    limits: CodecLimits,
    payload: &mut DecodedPayloadAccumulator,
) -> Result<Vec<Value>, QualifiedCodecError> {
    let num_columns = read_bounded_len(r, limits.max_row_columns, "row column count")?;

    if expected_columns > 0 && num_columns != expected_columns {
        return Err(QualifiedCodecPrimary::RowColumnMismatch {
            expected: expected_columns,
            actual: num_columns,
        }
        .into());
    }

    let mut budget = CodecBudget::new(limits);
    budget.charge_items_qualified(num_columns, std::mem::size_of::<Value>(), "row column")?;
    let mut row = Vec::new();
    for _ in 0..num_columns {
        let value = deserialize_value_with_budget(r, &mut budget, payload, 0)?;
        try_push(&mut row, value, "row columns")?;
    }
    Ok(row)
}

/// Decodes one already-framed row payload with an exact column count,
/// including the legitimate zero-column case.
#[cfg(feature = "spill")]
pub(crate) fn deserialize_framed_row_exact(
    payload: &[u8],
    expected_columns: usize,
    limits: CodecLimits,
) -> std::io::Result<Vec<Value>> {
    deserialize_framed_row_exact_inner(
        payload,
        expected_columns,
        limits,
        DecodedPayloadAccumulator::discarding(),
    )
    .map(|(row, _discarded)| row)
    .map_err(QualifiedCodecError::into_io)
}

/// Decodes one framed row and returns decoder-minted recursive payload proof.
///
/// The receipt excludes the top-level `Vec<Value>` allocation. Before either
/// value can escape, the decoder checks the observed top-level capacity plus
/// the recursively accumulated receipt against the broad construction
/// envelope for this exact frame. Existing public decode APIs retain their
/// established signatures and discard this additive proof.
///
/// # Errors
///
/// Returns an error for a malformed, truncated, or trailing-byte frame, a
/// column-count mismatch, checked retained-size overflow, allocation failure,
/// or an internal construction-envelope violation.
#[cfg(feature = "spill")]
pub(crate) fn deserialize_framed_row_exact_with_receipt(
    payload: &[u8],
    expected_columns: usize,
    limits: CodecLimits,
) -> std::io::Result<DecodedFramedRow> {
    deserialize_framed_row_exact_with_receipt_qualified(payload, expected_columns, limits)
        .map_err(QualifiedCodecError::into_io)
}

/// Decodes a framed row without allocating or detaching a diagnostic payload.
///
/// All malformed-frame, nested-value, resource, and allocation failures stay
/// in [`QualifiedCodecError`]. The compatibility sibling converts this typed
/// value into [`std::io::Error`] only after returning from the qualified path.
#[cfg(feature = "spill")]
pub(crate) fn deserialize_framed_row_exact_with_receipt_qualified(
    payload: &[u8],
    expected_columns: usize,
    limits: CodecLimits,
) -> Result<DecodedFramedRow, QualifiedCodecError> {
    let (row, retained) = deserialize_framed_row_exact_inner(
        payload,
        expected_columns,
        limits,
        DecodedPayloadAccumulator::tracking(),
    )?;

    let direct_bytes = row
        .capacity()
        .checked_mul(std::mem::size_of::<Value>())
        .ok_or(QualifiedCodecPrimary::PayloadRetainedOverflow {
            description: "top-level row capacity",
        })?;
    let retained_bytes = retained
        .retained_bytes
        .ok_or(QualifiedCodecPrimary::PayloadTrackingDisabled)?;
    let observed_bytes = direct_bytes.checked_add(retained_bytes).ok_or(
        QualifiedCodecPrimary::PayloadRetainedOverflow {
            description: "row and payload",
        },
    )?;
    let construction_envelope = conservative_decoded_row_retained_bytes(payload.len())
        .map_err(QualifiedCodecPrimary::ConstructionEnvelope)?;
    if observed_bytes > construction_envelope {
        return Err(QualifiedCodecPrimary::RetainedBeyondEnvelope {
            observed: observed_bytes,
            envelope: construction_envelope,
        }
        .into());
    }

    Ok(DecodedFramedRow {
        values: row,
        receipt: retained.into_receipt()?,
    })
}

#[cfg(feature = "spill")]
fn deserialize_framed_row_exact_inner(
    payload: &[u8],
    expected_columns: usize,
    limits: CodecLimits,
    mut retained: DecodedPayloadAccumulator,
) -> Result<(Vec<Value>, DecodedPayloadAccumulator), QualifiedCodecError> {
    let limits = limits.bounded_to_payload(payload.len());
    let Some(declared_bytes) = payload.get(..8) else {
        return Err(QualifiedCodecPrimary::FramedRowTooShort {
            actual: payload.len(),
        }
        .into());
    };
    let declared_bytes = [
        declared_bytes[0],
        declared_bytes[1],
        declared_bytes[2],
        declared_bytes[3],
        declared_bytes[4],
        declared_bytes[5],
        declared_bytes[6],
        declared_bytes[7],
    ];
    let encoded_declared = u64::from_le_bytes(declared_bytes);
    let declared = usize::try_from(encoded_declared).map_err(|_| {
        QualifiedCodecPrimary::LengthNotAddressable {
            description: "framed row column count",
            encoded: encoded_declared,
        }
    })?;
    if declared != expected_columns {
        return Err(QualifiedCodecPrimary::RowColumnMismatch {
            expected: expected_columns,
            actual: declared,
        }
        .into());
    }

    let mut payload_reader = payload;
    let mut reader = DecodeReader::framed(&mut payload_reader, payload.len());
    let row = deserialize_row_from_reader(&mut reader, expected_columns, limits, &mut retained)?;
    let remaining = reader.remaining_framed_bytes()?;
    if remaining != 0 {
        return Err(QualifiedCodecPrimary::TrailingFrameBytes { remaining }.into());
    }
    Ok((row, retained))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{
        DataChunk, QueryResourceContext, QueryResourceContextError, ValueVector,
    };
    use arcstr::ArcStr;
    use grafeo_common::memory::buffer::{
        BufferManager, BufferManagerConfig, MemoryGrantError, MemoryLimitScope,
    };
    use grafeo_common::types::INTERNAL_RDF_TAGGED_TERM_MARKER;
    use grafeo_common::types::LogicalType;
    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::io::Cursor;

    fn semantic_key(row: &[Value]) -> Vec<u8> {
        let limits = CodecLimits::default();
        let key = MeasuredSemanticKey::new(row, limits).unwrap();
        let measurement = key.measurement();
        let mut scratch = CounterSortScratch::new();
        scratch
            .prepare(
                measurement.counter_sort_entries,
                measurement.counter_sort_key_bytes,
            )
            .unwrap();
        let mut encoded = Vec::with_capacity(measurement.encoded_bytes);
        let written = key
            .serialize_with_prepared_scratch(&mut encoded, &mut scratch)
            .unwrap();
        assert_eq!(written, measurement.encoded_bytes);
        assert_eq!(written, encoded.len());
        encoded
    }

    #[test]
    fn semantic_key_codec_matches_exact_typed_equality_and_keeps_lossless_witnesses() {
        use grafeo_common::types::{Date, Duration, HashableValue, Time, Timestamp, ZonedDatetime};
        let zoned = |offset| {
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(123_456),
                offset,
            ))
        };
        let nested = |zero, offset| {
            Value::List(
                vec![
                    Value::Map(BTreeMap::from([("z".into(), Value::Float64(zero))]).into()),
                    Value::Path {
                        nodes: vec![zoned(offset)].into(),
                        edges: vec![Value::Float64(zero)].into(),
                    },
                ]
                .into(),
            )
        };
        let samples = vec![
            Value::Null,
            Value::Bool(false),
            Value::Int64(0),
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::Int64(4_607_182_418_800_017_408),
            Value::Float64(1.0),
            Value::from("1"),
            Value::Bytes(vec![1].into()),
            Value::Float64(f64::from_bits(0x7ff8_0000_0000_0001)),
            Value::Float64(f64::from_bits(0x7ff8_0000_0000_0001)),
            Value::Float64(f64::from_bits(0x7ff8_0000_0000_0002)),
            Value::Vector(vec![0.0].into()),
            Value::Vector(vec![-0.0].into()),
            Value::Timestamp(Timestamp::from_micros(123_456)),
            Value::Date(Date::from_days(123)),
            Value::Time(Time::from_nanos(123).unwrap()),
            Value::Duration(Duration::new(1, 2, 3)),
            Value::OnCounter {
                pos: Arc::new(HashMap::from([("a".into(), 1)])),
                neg: Arc::new(HashMap::from([("b".into(), 2)])),
            },
            Value::GCounter(Arc::new(HashMap::from([("a".into(), 1)]))),
            Value::RdfLiteral {
                lexical: "01".into(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
            },
            Value::RdfLiteral {
                lexical: "colour".into(),
                language: Some("en".into()),
                datatype: None,
            },
            zoned(0),
            zoned(3600),
            nested(-0.0, 3600),
            nested(0.0, 0),
        ];
        for left in &samples {
            for right in &samples {
                assert_eq!(
                    semantic_key(std::slice::from_ref(left))
                        == semantic_key(std::slice::from_ref(right)),
                    HashableValue::new(left.clone()) == HashableValue::new(right.clone()),
                    "key equality differs for {left:?} and {right:?}",
                );
            }
        }
        let row = [Value::Float64(-0.0), zoned(3600)];
        let mut witness = Vec::new();
        serialize_row_with_limits(&row, &mut witness, CodecLimits::default()).unwrap();
        assert_ne!(witness, semantic_key(&row));
        let decoded = deserialize_row(&mut witness.as_slice(), 2).unwrap();
        assert!(
            matches!(decoded[0], Value::Float64(value) if value.to_bits() == (-0.0f64).to_bits())
        );
        assert!(
            matches!(decoded[1], Value::ZonedDatetime(value) if value.offset_seconds() == 3600)
        );
    }

    #[test]
    fn semantic_key_codec_preflights_all_columns_and_counter_workspace_before_output() {
        let row = [Value::GCounter(std::sync::Arc::new(HashMap::from([
            ("second".into(), 2),
            ("first".into(), 1),
        ])))];
        let mut scratch = CounterSortScratch::new();
        let mut output = Vec::new();
        assert!(
            MeasuredSemanticKey::new(&row, CodecLimits::default())
                .unwrap()
                .serialize_with_prepared_scratch(&mut output, &mut scratch)
                .is_err()
        );
        assert!(output.is_empty());
        let reversed = [Value::GCounter(std::sync::Arc::new(HashMap::from([
            ("first".into(), 1),
            ("second".into(), 2),
        ])))];
        assert_eq!(semantic_key(&row), semantic_key(&reversed));
        let limits = CodecLimits {
            max_row_columns: 0,
            ..CodecLimits::default()
        };
        assert!(
            MeasuredSemanticKey::new(&row, limits)
                .and_then(|key| key.serialize_with_prepared_scratch(&mut output, &mut scratch))
                .is_err()
        );
        assert!(output.is_empty());
    }

    enum TestWriterFailure {
        Error,
        Panic,
    }

    struct FailOnExactWrite {
        target: &'static [u8],
        failure: TestWriterFailure,
        bytes: Vec<u8>,
    }

    impl FailOnExactWrite {
        fn new(target: &'static [u8], failure: TestWriterFailure) -> Self {
            Self {
                target,
                failure,
                bytes: Vec::new(),
            }
        }
    }

    impl std::io::Write for FailOnExactWrite {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes == self.target {
                match self.failure {
                    TestWriterFailure::Error => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "injected counter-key write failure",
                        ));
                    }
                    TestWriterFailure::Panic => panic!("injected counter-key write panic"),
                }
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn roundtrip_value(value: Value) -> Value {
        let mut buf = Vec::new();
        serialize_value(&value, &mut buf).unwrap();
        let mut cursor = Cursor::new(buf);
        deserialize_value(&mut cursor).unwrap()
    }

    #[cfg(feature = "spill")]
    fn decode_row_with_receipt(row: &[Value]) -> (Vec<Value>, DecodedPayloadReceipt, usize) {
        let mut encoded = Vec::new();
        serialize_row(row, &mut encoded).unwrap();
        let decoded = deserialize_framed_row_exact_with_receipt(
            &encoded,
            row.len(),
            CodecLimits::format_max(),
        )
        .unwrap();
        let (decoded, receipt) = decoded.into_parts();
        (decoded, receipt, encoded.len())
    }

    #[test]
    fn test_serialize_null() {
        let result = roundtrip_value(Value::Null);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_serialize_bool() {
        assert_eq!(roundtrip_value(Value::Bool(true)), Value::Bool(true));
        assert_eq!(roundtrip_value(Value::Bool(false)), Value::Bool(false));
    }

    #[test]
    fn test_serialize_int64() {
        assert_eq!(roundtrip_value(Value::Int64(0)), Value::Int64(0));
        assert_eq!(
            roundtrip_value(Value::Int64(i64::MAX)),
            Value::Int64(i64::MAX)
        );
        assert_eq!(
            roundtrip_value(Value::Int64(i64::MIN)),
            Value::Int64(i64::MIN)
        );
        assert_eq!(roundtrip_value(Value::Int64(-42)), Value::Int64(-42));
    }

    #[test]
    fn test_serialize_float64() {
        assert_eq!(roundtrip_value(Value::Float64(0.0)), Value::Float64(0.0));
        assert_eq!(
            roundtrip_value(Value::Float64(std::f64::consts::PI)),
            Value::Float64(std::f64::consts::PI)
        );
        // Note: NaN != NaN, so we test differently
        let nan_result = roundtrip_value(Value::Float64(f64::NAN));
        assert!(matches!(nan_result, Value::Float64(f) if f.is_nan()));
    }

    #[test]
    fn test_serialize_string() {
        let result = roundtrip_value(Value::String(ArcStr::from("hello world")));
        assert_eq!(result.as_str(), Some("hello world"));

        // Empty string
        let result = roundtrip_value(Value::String(ArcStr::from("")));
        assert_eq!(result.as_str(), Some(""));

        // Unicode
        let result = roundtrip_value(Value::String(ArcStr::from("héllo 世界 🌍")));
        assert_eq!(result.as_str(), Some("héllo 世界 🌍"));
    }

    #[test]
    fn test_serialize_bytes() {
        let data = vec![0u8, 1, 2, 255, 128];
        let result = roundtrip_value(Value::Bytes(Arc::from(data.clone())));
        assert_eq!(result.as_bytes(), Some(&data[..]));

        // Empty bytes
        let result = roundtrip_value(Value::Bytes(Arc::from(vec![])));
        assert_eq!(result.as_bytes(), Some(&[][..]));
    }

    #[test]
    fn test_serialize_timestamp() {
        let ts = grafeo_common::types::Timestamp::from_micros(1234567890);
        let result = roundtrip_value(Value::Timestamp(ts));
        assert_eq!(result.as_timestamp(), Some(ts));
    }

    #[test]
    fn test_serialize_list() {
        let list = Value::List(Arc::from(vec![
            Value::Int64(1),
            Value::String(ArcStr::from("two")),
            Value::Bool(true),
        ]));
        let result = roundtrip_value(list.clone());
        assert_eq!(result, list);

        // Nested list
        let nested = Value::List(Arc::from(vec![
            Value::List(Arc::from(vec![Value::Int64(1), Value::Int64(2)])),
            Value::List(Arc::from(vec![Value::Int64(3)])),
        ]));
        let result = roundtrip_value(nested.clone());
        assert_eq!(result, nested);

        // Empty list
        let empty = Value::List(Arc::from(vec![]));
        let result = roundtrip_value(empty.clone());
        assert_eq!(result, empty);
    }

    #[test]
    fn test_serialize_map() {
        let mut map = BTreeMap::new();
        map.insert(
            grafeo_common::types::PropertyKey::new("name"),
            Value::String(ArcStr::from("Alix")),
        );
        map.insert(
            grafeo_common::types::PropertyKey::new("age"),
            Value::Int64(30),
        );

        let value = Value::Map(Arc::new(map));
        let result = roundtrip_value(value.clone());
        assert_eq!(result, value);
    }

    #[test]
    fn test_serialize_row() {
        let row = vec![
            Value::Int64(1),
            Value::String(ArcStr::from("test")),
            Value::Bool(true),
            Value::Null,
        ];

        let mut buf = Vec::new();
        serialize_row(&row, &mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let result = deserialize_row(&mut cursor, 4).unwrap();
        assert_eq!(result, row);
    }

    #[test]
    fn row_measurement_matches_encoding_for_every_value_variant() {
        let mut property_map = BTreeMap::new();
        property_map.insert(
            grafeo_common::types::PropertyKey::new("nested"),
            Value::List(Arc::from([
                Value::String("value".into()),
                Value::Bytes(Arc::from([1_u8, 2, 3])),
            ])),
        );
        let grow_only = HashMap::from([
            ("replica-c".to_string(), 3),
            ("replica-a".to_string(), 1),
            ("replica-b".to_string(), 2),
        ]);
        let positive = HashMap::from([("replica-d".to_string(), 8), ("replica-b".to_string(), 5)]);
        let negative = HashMap::from([("replica-a".to_string(), 2)]);
        let timestamp = grafeo_common::types::Timestamp::from_micros(1_234_567);
        let time_without_offset =
            grafeo_common::types::Time::from_nanos(12_345).expect("test time is valid");
        let time_with_offset = time_without_offset.with_offset(3_600);
        let row = vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(-42),
            Value::Float64(3.5),
            Value::String("text".into()),
            Value::Bytes(Arc::from([0_u8, 1, 255])),
            Value::Timestamp(timestamp),
            Value::Date(grafeo_common::types::Date::from_days(20_000)),
            Value::Time(time_without_offset),
            Value::Time(time_with_offset),
            Value::Duration(grafeo_common::types::Duration::new(2, 3, 4)),
            Value::ZonedDatetime(grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                timestamp, -3_600,
            )),
            Value::List(Arc::from([Value::Int64(1), Value::Bool(false)])),
            Value::Map(Arc::new(property_map)),
            Value::Vector(Arc::from([1.0_f32, -2.5, 3.25])),
            Value::Path {
                nodes: Arc::from([Value::Int64(1), Value::Int64(2)]),
                edges: Arc::from([Value::String("edge".into())]),
            },
            Value::GCounter(Arc::new(grow_only)),
            Value::OnCounter {
                pos: Arc::new(positive),
                neg: Arc::new(negative),
            },
            Value::RdfLiteral {
                lexical: "colour".into(),
                language: Some("en-GB".into()),
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "18446744073709551616".into(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
            },
        ];

        let measurement =
            measure_serialized_row_with_limits(&row, CodecLimits::format_max()).unwrap();
        let mut encoded = Vec::new();
        let written = serialize_row(&row, &mut encoded).unwrap();

        assert_eq!(measurement.encoded_bytes, written);
        assert_eq!(measurement.encoded_bytes, encoded.len());
        assert_eq!(measurement.counter_sort_entries, 3);
        assert_eq!(measurement.counter_sort_key_bytes, 27);
    }

    #[test]
    fn conservative_decoded_row_bound_keeps_exact_direct_capacity_separate() {
        let shared_blob: Arc<[u8]> = Arc::from([1_u8, 2, 3, 4, 5]);
        let shared_text = ArcStr::from("shared text");
        let nested = Value::List(Arc::from([
            Value::String(shared_text.clone()),
            Value::Bytes(Arc::clone(&shared_blob)),
            Value::Map(Arc::new(BTreeMap::from([(
                grafeo_common::types::PropertyKey::new("nested"),
                Value::Path {
                    nodes: Arc::from([Value::String(shared_text.clone())]),
                    edges: Arc::from([Value::Bytes(Arc::clone(&shared_blob))]),
                },
            )]))),
            Value::RdfLiteral {
                lexical: shared_text.clone(),
                language: Some("en".into()),
                datatype: None,
            },
        ]));
        let row = vec![nested, Value::Bytes(Arc::clone(&shared_blob))];
        let measurement =
            measure_serialized_row_with_limits(&row, CodecLimits::format_max()).unwrap();
        let retained = conservative_decoded_row_retained_bytes(measurement.encoded_bytes).unwrap();

        let mut column = ValueVector::try_with_capacity(LogicalType::Any, row.len()).unwrap();
        for value in &row {
            column.try_push_value(value.clone()).unwrap();
        }
        let chunk = DataChunk::new(vec![column]);
        let exact_direct = chunk.observed_column_capacity_bytes().unwrap();
        let expected_direct =
            std::mem::size_of::<ValueVector>() + row.len() * std::mem::size_of::<Value>();

        assert_eq!(exact_direct, expected_direct);
        assert_eq!(
            retained,
            measurement.encoded_bytes
                * (std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>() + 16)
        );
        assert!(retained > exact_direct);
        assert_eq!(Arc::strong_count(&shared_blob), 5);
        assert_eq!(ArcStr::strong_count(&shared_text), Some(4));
        drop(row);
        assert_eq!(Arc::strong_count(&shared_blob), 4);
        assert_eq!(ArcStr::strong_count(&shared_text), Some(4));
        assert_eq!(
            chunk.column(0).unwrap().get_value(1),
            Some(Value::Bytes(Arc::clone(&shared_blob)))
        );
        drop(chunk);
        assert_eq!(Arc::strong_count(&shared_blob), 1);
        assert_eq!(ArcStr::strong_count(&shared_text), Some(1));
    }

    #[test]
    fn conservative_decoded_row_bound_covers_top_level_vec_growth_slack() {
        let row = vec![Value::Null; 17];
        let mut encoded = Vec::new();
        serialize_row(&row, &mut encoded).unwrap();
        let retained = conservative_decoded_row_retained_bytes(encoded.len()).unwrap();
        let decoded = deserialize_row(&mut Cursor::new(&encoded), row.len()).unwrap();
        let observed_top_level =
            decoded.capacity() * std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>();
        let previous_envelope =
            encoded.len() * std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>();

        assert_eq!(encoded.len(), 25);
        assert!(
            observed_top_level > previous_envelope,
            "the fixture must exercise amortized Vec capacity beyond the old envelope"
        );
        assert!(retained >= observed_top_level);
    }

    #[test]
    fn conservative_decoded_row_bound_covers_many_nested_vec_growth_shapes() {
        let nested = Value::List(Arc::from(vec![Value::Null; 17]));
        let row = vec![nested; 17];
        let measurement =
            measure_serialized_row_with_limits(&row, CodecLimits::format_max()).unwrap();
        let retained = conservative_decoded_row_retained_bytes(measurement.encoded_bytes).unwrap();
        let per_wire_byte = std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>() + 16;

        assert_eq!(measurement.encoded_bytes, 450);
        assert_eq!(retained, 450 * per_wire_byte);

        let mut encoded = Vec::new();
        serialize_row(&row, &mut encoded).unwrap();
        let decoded = deserialize_row(&mut Cursor::new(encoded), row.len()).unwrap();
        assert_eq!(decoded.len(), 17);
        assert!(
            decoded
                .iter()
                .all(|value| value.as_list().unwrap().len() == 17)
        );
    }

    #[test]
    fn conservative_decoded_row_bound_reports_multiplication_overflow() {
        let per_wire_byte = std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>() + 16;
        let encoded_bytes = usize::MAX / per_wire_byte + 1;

        let error = conservative_decoded_row_retained_bytes(encoded_bytes).unwrap_err();

        assert_eq!(
            error,
            MemoryGrantError::ArithmeticOverflow {
                current_bytes: encoded_bytes,
                additional_bytes: per_wire_byte,
            }
        );
    }

    #[test]
    fn conservative_decoded_row_bound_drives_atomic_grant_denial() {
        // `[columns: u64][string tag][length: u64][abc]`.
        let encoded_bytes = 8 + 1 + 8 + 3;
        let retained = conservative_decoded_row_retained_bytes(encoded_bytes).unwrap();
        let mut config = BufferManagerConfig::with_budget(retained - 1);
        config.soft_limit_fraction = 0.5;
        config.evict_limit_fraction = 0.75;
        config.hard_limit_fraction = 1.0;
        let manager = BufferManager::new(config);
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();

        let error = context.try_allocate(retained).unwrap_err();

        assert_eq!(
            error,
            QueryResourceContextError::Memory(MemoryGrantError::LimitExceeded {
                scope: MemoryLimitScope::Query,
                requested_bytes: retained,
                limit_bytes: retained - 1,
            })
        );
        assert_eq!(context.query_stats().allocated_bytes, 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_receipt_covers_every_recursive_heap_variant() {
        let variants = [
            (
                Value::String(ArcStr::from("string-payload")),
                "string-payload".len(),
            ),
            (Value::Bytes(Arc::from([1_u8, 2, 3, 4, 5])), 5),
            (
                Value::List(Arc::from([
                    Value::String(ArcStr::from("nested")),
                    Value::Bytes(Arc::from([7_u8, 8, 9])),
                ])),
                2 * std::mem::size_of::<Value>() + "nested".len() + 3,
            ),
            (
                Value::Map(Arc::new(BTreeMap::from([(
                    grafeo_common::types::PropertyKey::new("property-key"),
                    Value::List(Arc::from([Value::String(ArcStr::from("map-value"))])),
                )]))),
                std::mem::size_of::<(grafeo_common::types::PropertyKey, Value)>()
                    + std::mem::size_of::<Value>()
                    + "property-key".len()
                    + "map-value".len(),
            ),
            (
                Value::Vector(Arc::from([1.0_f32, 2.0, 3.0, 4.0])),
                4 * std::mem::size_of::<f32>(),
            ),
            (
                Value::Path {
                    nodes: Arc::from([
                        Value::String(ArcStr::from("node-a")),
                        Value::String(ArcStr::from("node-b")),
                    ]),
                    edges: Arc::from([Value::Bytes(Arc::from([4_u8, 5, 6]))]),
                },
                3 * std::mem::size_of::<Value>() + "node-a".len() + "node-b".len() + 3,
            ),
            (
                Value::GCounter(Arc::new(HashMap::from([
                    ("replica-a".to_string(), 1),
                    ("replica-b".to_string(), 2),
                ]))),
                2 * std::mem::size_of::<(String, u64)>() + "replica-a".len() + "replica-b".len(),
            ),
            (
                Value::OnCounter {
                    pos: Arc::new(HashMap::from([("positive".to_string(), 4)])),
                    neg: Arc::new(HashMap::from([("negative".to_string(), 3)])),
                },
                2 * std::mem::size_of::<(String, u64)>() + "positive".len() + "negative".len(),
            ),
            (
                Value::RdfLiteral {
                    lexical: ArcStr::from("lexical"),
                    language: Some(ArcStr::from("en-GB")),
                    datatype: Some(ArcStr::from("urn:datatype")),
                },
                "lexical".len() + "en-GB".len() + "urn:datatype".len(),
            ),
        ];

        for (value, minimum_payload_bytes) in variants {
            let row = [value];
            let (decoded, receipt, encoded_bytes) = decode_row_with_receipt(&row);
            let direct_bytes = decoded.capacity() * std::mem::size_of::<Value>();
            let construction_envelope =
                conservative_decoded_row_retained_bytes(encoded_bytes).unwrap();

            assert_eq!(decoded, row);
            assert!(
                receipt.retained_bytes() >= minimum_payload_bytes,
                "receipt omitted retained storage for {}",
                row[0].type_name()
            );
            assert!(direct_bytes + receipt.retained_bytes() <= construction_envelope);
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_receipt_counts_equal_source_payloads_as_independent_allocations() {
        let shared_text = ArcStr::from("independently decoded text");
        let shared_blob: Arc<[u8]> = Arc::from([3_u8; 257]);
        let shared_nested: Arc<[Value]> =
            Arc::from([Value::String(shared_text), Value::Bytes(shared_blob)]);
        let row = [
            Value::List(Arc::clone(&shared_nested)),
            Value::List(shared_nested),
        ];

        let (decoded, receipt, _) = decode_row_with_receipt(&row);
        let (Value::List(left), Value::List(right)) = (&decoded[0], &decoded[1]) else {
            panic!("fixture must decode as two lists");
        };
        let (Value::String(left_text), Value::String(right_text)) = (&left[0], &right[0]) else {
            panic!("fixture must retain nested strings");
        };
        let (Value::Bytes(left_blob), Value::Bytes(right_blob)) = (&left[1], &right[1]) else {
            panic!("fixture must retain nested byte strings");
        };

        assert!(!Arc::ptr_eq(left, right));
        assert!(!ArcStr::ptr_eq(left_text, right_text));
        assert!(!Arc::ptr_eq(left_blob, right_blob));
        assert!(
            receipt.retained_bytes()
                >= 2 * (2 * std::mem::size_of::<Value>()
                    + "independently decoded text".len()
                    + 257)
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_receipt_excludes_top_level_vec_capacity() {
        let heap_value = Value::Bytes(Arc::from([11_u8; 31]));
        let compact_row = [heap_value.clone()];
        let mut wide_row = vec![heap_value];
        wide_row.extend(std::iter::repeat_n(Value::Null, 16));

        let (compact, compact_receipt, compact_encoded) = decode_row_with_receipt(&compact_row);
        let (wide, wide_receipt, wide_encoded) = decode_row_with_receipt(&wide_row);
        let compact_direct = compact.capacity() * std::mem::size_of::<Value>();
        let wide_direct = wide.capacity() * std::mem::size_of::<Value>();

        assert!(wide_direct > compact_direct);
        assert_eq!(
            compact_receipt.retained_bytes(),
            wide_receipt.retained_bytes(),
            "top-level row growth must remain separate from the payload receipt"
        );
        assert!(
            compact_direct + compact_receipt.retained_bytes()
                <= conservative_decoded_row_retained_bytes(compact_encoded).unwrap()
        );
        assert!(
            wide_direct + wide_receipt.retained_bytes()
                <= conservative_decoded_row_retained_bytes(wide_encoded).unwrap()
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_receipt_is_near_linear_for_large_strings_and_blobs() {
        for value in [
            Value::String(ArcStr::from("s".repeat(128 * 1024))),
            Value::Bytes(Arc::from(vec![0xabu8; 128 * 1024])),
        ] {
            let row = [value];
            let (decoded, receipt, encoded_bytes) = decode_row_with_receipt(&row);
            let broad = conservative_decoded_row_retained_bytes(encoded_bytes).unwrap();
            let direct = decoded.capacity() * std::mem::size_of::<Value>();

            assert!(receipt.retained_bytes() >= 128 * 1024);
            assert!(receipt.retained_bytes() < broad / 16);
            assert!(direct + receipt.retained_bytes() <= broad);
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_receipt_accepts_adversarial_recursive_shapes_within_envelope() {
        let flat = Value::List(Arc::from(vec![Value::Null; 4_097]));
        let mut deep = Value::Bytes(Arc::from([19_u8; 97]));
        for _ in 0..64 {
            deep = Value::List(Arc::from([deep]));
        }
        let path = Value::Path {
            nodes: Arc::from([deep, Value::String(ArcStr::from("terminal-node"))]),
            edges: Arc::from([Value::List(Arc::from(vec![Value::Null; 257]))]),
        };
        let map = Value::Map(Arc::new(
            (0..257)
                .map(|index| {
                    (
                        grafeo_common::types::PropertyKey::new(format!(
                            "key-{index:04}-{}",
                            "k".repeat(23)
                        )),
                        Value::String(ArcStr::from(format!(
                            "value-{index:04}-{}",
                            "v".repeat(257)
                        ))),
                    )
                })
                .collect(),
        ));
        let row = [flat, path, map];

        let (decoded, receipt, encoded_bytes) = decode_row_with_receipt(&row);
        let direct = decoded.capacity() * std::mem::size_of::<Value>();
        let broad = conservative_decoded_row_retained_bytes(encoded_bytes).unwrap();

        assert_eq!(decoded, row);
        assert!(direct + receipt.retained_bytes() <= broad);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_accumulator_reports_checked_multiply_and_add_overflow() {
        let mut multiply = DecodedPayloadAccumulator::tracking();
        let multiply_error = multiply
            .include_allocation(usize::MAX, 2, "test allocation")
            .unwrap_err();
        let mut add = DecodedPayloadAccumulator {
            retained_bytes: Some(usize::MAX),
        };
        let add_error = add.include_allocation(1, 1, "test allocation").unwrap_err();

        assert_eq!(multiply_error.kind(), std::io::ErrorKind::OutOfMemory);
        assert_eq!(add_error.kind(), std::io::ErrorKind::OutOfMemory);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_receipt_never_escapes_malformed_framed_decode() {
        let row = [Value::List(Arc::from([
            Value::String(ArcStr::from("complete")),
            Value::Bytes(Arc::from([1_u8; 73])),
        ]))];
        let mut encoded = Vec::new();
        serialize_row(&row, &mut encoded).unwrap();
        let mut trailing = encoded.clone();
        trailing.push(0xff);
        let truncated = &encoded[..encoded.len() - 1];

        let trailing_error =
            deserialize_framed_row_exact_with_receipt(&trailing, 1, CodecLimits::format_max())
                .unwrap_err();
        let truncated_error =
            deserialize_framed_row_exact_with_receipt(truncated, 1, CodecLimits::format_max())
                .unwrap_err();
        let (_, first, _) = decode_row_with_receipt(&row);
        let (_, retry, _) = decode_row_with_receipt(&row);

        assert_eq!(trailing_error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(truncated_error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(first.retained_bytes(), retry.retained_bytes());
    }

    #[cfg(feature = "spill")]
    #[test]
    fn qualified_framed_decode_reports_eight_byte_truncation_without_io_conversion() {
        let payload = 1_u64.to_le_bytes();

        let error = deserialize_framed_row_exact_with_receipt_qualified(
            &payload,
            1,
            CodecLimits::format_max(),
        )
        .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            error.class(),
            QualifiedCodecErrorClass::TruncatedFrame {
                requested: 1,
                remaining: 0,
            }
        );
    }

    #[cfg(feature = "spill")]
    #[test]
    fn qualified_framed_decode_keeps_nested_length_diagnostics_typed() {
        const DECLARED_BYTES: u64 = 4 * 1024 * 1024;
        let declared_bytes =
            usize::try_from(DECLARED_BYTES).expect("four-megabyte test declaration fits usize");
        let mut payload = 1_u64.to_le_bytes().to_vec();
        payload.push(TAG_STRING);
        payload.extend_from_slice(&DECLARED_BYTES.to_le_bytes());

        let error = deserialize_framed_row_exact_with_receipt_qualified(
            &payload,
            1,
            CodecLimits::new(
                declared_bytes
                    .checked_add(1024)
                    .expect("test codec limit fits usize"),
                16,
                1,
                8,
            ),
        )
        .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            error.class(),
            QualifiedCodecErrorClass::FramedLengthExceeds {
                description: "string",
                length: declared_bytes,
                remaining: 0,
            }
        );
    }

    #[test]
    fn qualified_allocation_failure_converts_to_compatibility_out_of_memory_kind() {
        let error = QualifiedCodecError::from(QualifiedCodecPrimary::SharedAllocation {
            description: "test payload",
        })
        .into_io();

        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn decoded_payload_capabilities_are_not_clone() {
        trait AmbiguousIfClone<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        impl<T: Clone> AmbiguousIfClone<u8> for T {}

        let _ = <DecodedPayloadReceipt as AmbiguousIfClone<_>>::marker;
        let _ = <DecodedFramedRow as AmbiguousIfClone<_>>::marker;
    }

    #[test]
    fn row_measurement_tracks_largest_nested_counter_workspace() {
        let nested = Value::List(Arc::from([
            Value::GCounter(Arc::new(HashMap::from([("one".to_string(), 1)]))),
            Value::Map(Arc::new(BTreeMap::from([(
                grafeo_common::types::PropertyKey::new("counter"),
                Value::OnCounter {
                    pos: Arc::new(HashMap::from([
                        ("a".to_string(), 1),
                        ("b".to_string(), 2),
                        ("c".to_string(), 3),
                        ("d".to_string(), 4),
                    ])),
                    neg: Arc::new(HashMap::from([
                        ("long-negative-key-a".to_string(), 1),
                        ("long-negative-key-b".to_string(), 2),
                    ])),
                },
            )]))),
        ]));

        let row = [nested];
        let measurement =
            measure_serialized_row_with_limits(&row, CodecLimits::format_max()).unwrap();

        assert_eq!(measurement.counter_sort_entries, 4);
        assert_eq!(measurement.counter_sort_key_bytes, 38);

        let mut scratch = CounterSortScratch::new();
        scratch
            .prepare(
                measurement.counter_sort_entries,
                measurement.counter_sort_key_bytes,
            )
            .unwrap();
        let mut encoded = Vec::new();
        serialize_row_with_prepared_scratch(
            &row,
            &mut encoded,
            CodecLimits::format_max(),
            &mut scratch,
        )
        .unwrap();
        assert_eq!(deserialize_row(&mut Cursor::new(encoded), 1).unwrap(), row);
        assert_eq!(scratch.entry_len(), 0);
        assert_eq!(scratch.key_len(), 0);
        assert_eq!(scratch.write_growths(), 0);
    }

    #[test]
    fn row_measurement_applies_the_same_limits_as_serialization() {
        let row = [Value::Bytes(Arc::from([1_u8, 2, 3, 4, 5]))];
        let limits = CodecLimits::new(4, 16, 4, 8);

        let measured = measure_serialized_row_with_limits(&row, limits).unwrap_err();
        let mut encoded = Vec::new();
        let serialized = serialize_row_with_limits(&row, &mut encoded, limits).unwrap_err();

        assert_eq!(measured.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(measured.to_string(), serialized.to_string());
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_serialize_row_column_count_check() {
        let row = vec![Value::Int64(1), Value::Int64(2)];

        let mut buf = Vec::new();
        serialize_row(&row, &mut buf).unwrap();

        // Wrong expected column count
        let mut cursor = Cursor::new(buf.clone());
        let result = deserialize_row(&mut cursor, 3);
        assert!(result.is_err());

        // Skip check with 0
        let mut cursor = Cursor::new(buf);
        let result = deserialize_row(&mut cursor, 0).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_serialize_multiple_rows() {
        let rows = vec![
            vec![Value::Int64(1), Value::String(ArcStr::from("a"))],
            vec![Value::Int64(2), Value::String(ArcStr::from("b"))],
            vec![Value::Int64(3), Value::String(ArcStr::from("c"))],
        ];

        let mut buf = Vec::new();
        for row in &rows {
            serialize_row(row, &mut buf).unwrap();
        }

        let mut cursor = Cursor::new(buf);
        for expected in &rows {
            let result = deserialize_row(&mut cursor, 2).unwrap();
            assert_eq!(&result, expected);
        }
    }

    #[test]
    fn test_serialize_gcounter_roundtrip() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("replica-1".to_string(), 42u64);
        counts.insert("replica-2".to_string(), 17u64);
        let v = Value::GCounter(Arc::new(counts));
        let result = roundtrip_value(v.clone());
        assert_eq!(result, v);
    }

    #[test]
    fn test_serialize_gcounter_empty() {
        let v = Value::GCounter(Arc::new(std::collections::HashMap::new()));
        let result = roundtrip_value(v.clone());
        assert_eq!(result, v);
    }

    #[test]
    fn test_serialize_pncounter_roundtrip() {
        let mut pos = std::collections::HashMap::new();
        pos.insert("node-a".to_string(), 10u64);
        pos.insert("node-b".to_string(), 5u64);
        let mut neg = std::collections::HashMap::new();
        neg.insert("node-a".to_string(), 3u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        let result = roundtrip_value(v.clone());
        assert_eq!(result, v);
    }

    #[test]
    fn test_serialization_size() {
        // Verify expected sizes
        let mut buf = Vec::new();

        // Null: 1 byte (tag only)
        serialize_value(&Value::Null, &mut buf).unwrap();
        assert_eq!(buf.len(), 1);
        buf.clear();

        // Bool: 2 bytes (tag + value)
        serialize_value(&Value::Bool(true), &mut buf).unwrap();
        assert_eq!(buf.len(), 2);
        buf.clear();

        // Int64: 9 bytes (tag + 8)
        serialize_value(&Value::Int64(42), &mut buf).unwrap();
        assert_eq!(buf.len(), 9);
        buf.clear();

        // String "hi": 11 bytes (tag + 8 length + 2)
        serialize_value(&Value::String(ArcStr::from("hi")), &mut buf).unwrap();
        assert_eq!(buf.len(), 11);
    }

    #[test]
    fn all_current_value_variants_roundtrip_losslessly() {
        let mut property_map = BTreeMap::new();
        property_map.insert(
            grafeo_common::types::PropertyKey::new("answer"),
            Value::Int64(42),
        );
        let mut grow_only = HashMap::new();
        grow_only.insert("replica-b".to_string(), 2);
        grow_only.insert("replica-a".to_string(), 1);
        let mut positive = HashMap::new();
        positive.insert("replica-b".to_string(), 7);
        positive.insert("replica-a".to_string(), 5);
        let mut negative = HashMap::new();
        negative.insert("replica-a".to_string(), 3);

        let timestamp = grafeo_common::types::Timestamp::from_micros(1_234_567);
        let time = grafeo_common::types::Time::from_nanos(12_345)
            .expect("test time is valid")
            .with_offset(3_600);
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(-42),
            Value::Float64(3.5),
            Value::String("text".into()),
            Value::Bytes(Arc::from([0_u8, 1, 255])),
            Value::Timestamp(timestamp),
            Value::Date(grafeo_common::types::Date::from_days(20_000)),
            Value::Time(time),
            Value::Duration(grafeo_common::types::Duration::new(2, 3, 4)),
            Value::ZonedDatetime(grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                timestamp, -3_600,
            )),
            Value::List(Arc::from([Value::Int64(1), Value::Bool(false)])),
            Value::Map(Arc::new(property_map)),
            Value::Vector(Arc::from([1.0_f32, -2.5, 3.25])),
            Value::Path {
                nodes: Arc::from([Value::Int64(1), Value::Int64(2)]),
                edges: Arc::from([Value::String("edge".into())]),
            },
            Value::GCounter(Arc::new(grow_only)),
            Value::OnCounter {
                pos: Arc::new(positive),
                neg: Arc::new(negative),
            },
            Value::RdfLiteral {
                lexical: "plain".into(),
                language: None,
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "colour".into(),
                language: Some("en".into()),
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "18446744073709551616".into(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
            },
        ];

        for value in values {
            assert_eq!(roundtrip_value(value.clone()), value);
        }
    }

    #[test]
    fn nested_rdf_literals_roundtrip_losslessly() {
        let mut map = BTreeMap::new();
        map.insert(
            grafeo_common::types::PropertyKey::new("term"),
            Value::RdfLiteral {
                lexical: "bonjour".into(),
                language: Some("fr".into()),
                datatype: None,
            },
        );
        let tagged = Value::List(Arc::from([
            Value::Map(Arc::new(map)),
            Value::String("\"bonjour\"@fr".into()),
            Value::String(INTERNAL_RDF_TAGGED_TERM_MARKER.into()),
        ]));

        assert_eq!(roundtrip_value(tagged.clone()), tagged);
    }

    #[test]
    fn gcounter_encoding_is_deterministic() {
        let first = Value::GCounter(Arc::new(HashMap::from([
            ("replica-c".to_string(), 3),
            ("replica-a".to_string(), 1),
            ("replica-b".to_string(), 2),
        ])));
        let second = Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-c".to_string(), 3),
            ("replica-a".to_string(), 1),
        ])));
        let mut first_bytes = Vec::new();
        let mut second_bytes = Vec::new();

        serialize_value(&first, &mut first_bytes).unwrap();
        serialize_value(&second, &mut second_bytes).unwrap();

        assert_eq!(first_bytes, second_bytes);
    }

    #[test]
    fn pncounter_encoding_is_deterministic() {
        let first = Value::OnCounter {
            pos: Arc::new(HashMap::from([
                ("replica-c".to_string(), 3),
                ("replica-a".to_string(), 1),
                ("replica-b".to_string(), 2),
            ])),
            neg: Arc::new(HashMap::from([
                ("replica-b".to_string(), 8),
                ("replica-a".to_string(), 5),
            ])),
        };
        let second = Value::OnCounter {
            pos: Arc::new(HashMap::from([
                ("replica-b".to_string(), 2),
                ("replica-c".to_string(), 3),
                ("replica-a".to_string(), 1),
            ])),
            neg: Arc::new(HashMap::from([
                ("replica-a".to_string(), 5),
                ("replica-b".to_string(), 8),
            ])),
        };
        let mut first_bytes = Vec::new();
        let mut second_bytes = Vec::new();

        serialize_value(&first, &mut first_bytes).unwrap();
        serialize_value(&second, &mut second_bytes).unwrap();

        assert_eq!(first_bytes, second_bytes);
    }

    #[test]
    fn prepared_counter_scratch_is_v1_byte_exact_and_reusable() {
        let first = Value::GCounter(Arc::new(HashMap::from([
            ("z".to_string(), 2),
            ("é".to_string(), 3),
            ("a".to_string(), 1),
            ("aa".to_string(), 4),
            ("\0".to_string(), 5),
            (String::new(), 6),
        ])));
        let second = Value::GCounter(Arc::new(HashMap::from([
            ("é".to_string(), 3),
            (String::new(), 6),
            ("aa".to_string(), 4),
            ("a".to_string(), 1),
            ("\0".to_string(), 5),
            ("z".to_string(), 2),
        ])));
        let mut expected = vec![TAG_GCOUNTER];
        expected.extend_from_slice(&6_u64.to_le_bytes());
        append_counter_entry(&mut expected, "", 6);
        append_counter_entry(&mut expected, "\0", 5);
        append_counter_entry(&mut expected, "a", 1);
        append_counter_entry(&mut expected, "aa", 4);
        append_counter_entry(&mut expected, "z", 2);
        append_counter_entry(&mut expected, "é", 3);

        let mut scratch = CounterSortScratch::new();
        let observed = scratch.prepare(6, 7).unwrap();
        let entry_pointer = scratch.entry_pointer();
        let key_pointer = scratch.key_pointer();

        for value in [&first, &second] {
            let mut encoded = Vec::new();
            serialize_value_unchecked(value, &mut encoded, &mut scratch).unwrap();
            assert_eq!(encoded, expected);
            assert_eq!(scratch.entry_len(), 0);
            assert_eq!(scratch.key_len(), 0);
            assert_eq!(scratch.entry_capacity(), observed.entry_capacity);
            assert_eq!(scratch.key_capacity(), observed.key_capacity);
            assert_eq!(scratch.entry_pointer(), entry_pointer);
            assert_eq!(scratch.key_pointer(), key_pointer);
            assert_eq!(scratch.write_growths(), 0);
        }
    }

    #[test]
    fn underprepared_counter_scratch_refuses_before_row_header() {
        let row = [Value::GCounter(Arc::new(HashMap::from([
            ("replica-a".to_string(), 1),
            ("replica-b".to_string(), 2),
        ])))];
        let mut scratch = CounterSortScratch::new();
        scratch.prepare(1, 18).unwrap();
        let mut encoded = Vec::new();

        let error = serialize_row_with_prepared_scratch(
            &row,
            &mut encoded,
            CodecLimits::format_max(),
            &mut scratch,
        )
        .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::OutOfMemory);
        assert!(encoded.is_empty());
        assert_eq!(scratch.entry_len(), 0);
        assert_eq!(scratch.key_len(), 0);
    }

    #[test]
    fn impossible_counter_preparation_preserves_reusable_capacity() {
        let mut scratch = CounterSortScratch::new();
        let observed = scratch.prepare(4, 32).unwrap();
        let entry_pointer = scratch.entry_pointer();
        let key_pointer = scratch.key_pointer();

        let entries_error = scratch.prepare(usize::MAX, 32).unwrap_err();
        let keys_error = scratch.prepare(4, usize::MAX).unwrap_err();

        assert_eq!(entries_error.kind(), std::io::ErrorKind::OutOfMemory);
        assert_eq!(keys_error.kind(), std::io::ErrorKind::OutOfMemory);
        assert_eq!(scratch.entry_capacity(), observed.entry_capacity);
        assert_eq!(scratch.key_capacity(), observed.key_capacity);
        assert_eq!(scratch.entry_pointer(), entry_pointer);
        assert_eq!(scratch.key_pointer(), key_pointer);
        assert_eq!(scratch.entry_len(), 0);
        assert_eq!(scratch.key_len(), 0);

        let value = Value::GCounter(Arc::new(HashMap::from([("replica-a".to_string(), 1)])));
        let mut encoded = Vec::new();
        serialize_value_unchecked(&value, &mut encoded, &mut scratch).unwrap();
        assert_eq!(deserialize_value(&mut Cursor::new(encoded)).unwrap(), value);
    }

    #[test]
    fn counter_writer_error_clears_scratch_for_deterministic_retry() {
        let value = Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])));
        let mut scratch = CounterSortScratch::new();
        let observed = scratch.prepare(2, 18).unwrap();
        let entry_pointer = scratch.entry_pointer();
        let key_pointer = scratch.key_pointer();
        let mut writer = FailOnExactWrite::new(b"replica-a", TestWriterFailure::Error);

        let error = serialize_value_unchecked(&value, &mut writer, &mut scratch).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(scratch.entry_len(), 0);
        assert_eq!(scratch.key_len(), 0);
        assert_eq!(scratch.entry_capacity(), observed.entry_capacity);
        assert_eq!(scratch.key_capacity(), observed.key_capacity);
        assert_eq!(scratch.entry_pointer(), entry_pointer);
        assert_eq!(scratch.key_pointer(), key_pointer);
        assert_eq!(scratch.write_growths(), 0);

        let mut retry = Vec::new();
        serialize_value_unchecked(&value, &mut retry, &mut scratch).unwrap();
        let mut expected = vec![TAG_GCOUNTER];
        expected.extend_from_slice(&2_u64.to_le_bytes());
        append_counter_entry(&mut expected, "replica-a", 1);
        append_counter_entry(&mut expected, "replica-b", 2);
        assert_eq!(retry, expected);
    }

    #[test]
    fn caught_counter_writer_panic_clears_scratch_for_deterministic_retry() {
        let value = Value::GCounter(Arc::new(HashMap::from([
            ("replica-b".to_string(), 2),
            ("replica-a".to_string(), 1),
        ])));
        let mut scratch = CounterSortScratch::new();
        let observed = scratch.prepare(2, 18).unwrap();
        let entry_pointer = scratch.entry_pointer();
        let key_pointer = scratch.key_pointer();
        let mut writer = FailOnExactWrite::new(b"replica-a", TestWriterFailure::Panic);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            serialize_value_unchecked(&value, &mut writer, &mut scratch)
        }));

        assert!(panic.is_err());
        assert_eq!(scratch.entry_len(), 0);
        assert_eq!(scratch.key_len(), 0);
        assert_eq!(scratch.entry_capacity(), observed.entry_capacity);
        assert_eq!(scratch.key_capacity(), observed.key_capacity);
        assert_eq!(scratch.entry_pointer(), entry_pointer);
        assert_eq!(scratch.key_pointer(), key_pointer);
        assert_eq!(scratch.write_growths(), 0);

        let mut retry = Vec::new();
        serialize_value_unchecked(&value, &mut retry, &mut scratch).unwrap();
        let mut expected = vec![TAG_GCOUNTER];
        expected.extend_from_slice(&2_u64.to_le_bytes());
        append_counter_entry(&mut expected, "replica-a", 1);
        append_counter_entry(&mut expected, "replica-b", 2);
        assert_eq!(retry, expected);
    }

    #[test]
    fn deserialize_value_rejects_unknown_tag() {
        let error = deserialize_value(&mut Cursor::new([u8::MAX])).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_value_rejects_truncated_payload() {
        let mut encoded = vec![TAG_STRING];
        encoded.extend_from_slice(&3_u64.to_le_bytes());
        encoded.extend_from_slice(b"ab");

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn framed_scalar_lengths_fail_before_growth_when_payload_is_short() {
        const DECLARED_BYTES: u64 = 4 * 1024 * 1024;
        let declared_bytes =
            usize::try_from(DECLARED_BYTES).expect("four-megabyte test declaration fits usize");

        for tag in [TAG_STRING, TAG_BYTES] {
            let mut payload = 1_u64.to_le_bytes().to_vec();
            payload.push(tag);
            payload.extend_from_slice(&DECLARED_BYTES.to_le_bytes());

            let error = deserialize_framed_row_exact(
                &payload,
                1,
                CodecLimits::new(
                    declared_bytes
                        .checked_add(1024)
                        .expect("test codec limit fits usize"),
                    16,
                    1,
                    8,
                ),
            )
            .unwrap_err();

            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert!(
                error
                    .to_string()
                    .contains("exceeds 0 remaining framed payload bytes"),
                "unexpected framed-length error: {error}"
            );
        }
    }

    #[cfg(feature = "spill")]
    #[test]
    fn framed_nested_string_fields_share_the_remaining_byte_guard() {
        const DECLARED_BYTES: u64 = 4 * 1024 * 1024;
        let declared_bytes =
            usize::try_from(DECLARED_BYTES).expect("four-megabyte test declaration fits usize");
        let row_header = 1_u64.to_le_bytes();
        let declared = DECLARED_BYTES.to_le_bytes();
        let one_item = 1_u64.to_le_bytes();
        let empty = 0_u64.to_le_bytes();

        let mut map_key = row_header.to_vec();
        map_key.push(TAG_MAP);
        map_key.extend_from_slice(&one_item);
        map_key.extend_from_slice(&declared);

        let mut counter_key = row_header.to_vec();
        counter_key.push(TAG_GCOUNTER);
        counter_key.extend_from_slice(&one_item);
        counter_key.extend_from_slice(&declared);

        let mut rdf_lexical = row_header.to_vec();
        rdf_lexical.push(TAG_RDF_LITERAL);
        rdf_lexical.extend_from_slice(&declared);

        let mut rdf_language = row_header.to_vec();
        rdf_language.push(TAG_RDF_LITERAL);
        rdf_language.extend_from_slice(&empty);
        rdf_language.push(1);
        rdf_language.extend_from_slice(&declared);

        let mut rdf_datatype = row_header.to_vec();
        rdf_datatype.push(TAG_RDF_LITERAL);
        rdf_datatype.extend_from_slice(&empty);
        rdf_datatype.push(0);
        rdf_datatype.push(1);
        rdf_datatype.extend_from_slice(&declared);

        for (field, payload) in [
            ("map key", map_key),
            ("counter key", counter_key),
            ("RDF lexical", rdf_lexical),
            ("RDF language", rdf_language),
            ("RDF datatype", rdf_datatype),
        ] {
            let error = deserialize_framed_row_exact(
                &payload,
                1,
                CodecLimits::new(
                    declared_bytes
                        .checked_add(1024)
                        .expect("test codec limit fits usize"),
                    16,
                    1,
                    8,
                ),
            )
            .unwrap_err();

            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData, "{field}");
            assert!(
                error
                    .to_string()
                    .contains("exceeds 0 remaining framed payload bytes"),
                "unexpected {field} error: {error}"
            );
        }
    }

    #[test]
    fn deserialize_value_with_explicit_limits_rejects_oversized_byte_length() {
        let mut encoded = vec![TAG_STRING];
        encoded.extend_from_slice(&5_u64.to_le_bytes());
        let limits = CodecLimits::new(4, 16, 4, 8);

        let error = deserialize_value_with_limits(&mut Cursor::new(encoded), limits).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_value_with_explicit_limits_rejects_oversized_collection_length() {
        let mut encoded = vec![TAG_LIST];
        encoded.extend_from_slice(&3_u64.to_le_bytes());
        let limits = CodecLimits::new(1024, 2, 4, 8);

        let error = deserialize_value_with_limits(&mut Cursor::new(encoded), limits).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_row_with_explicit_limits_rejects_oversized_column_count() {
        let encoded = 3_u64.to_le_bytes();
        let limits = CodecLimits::new(1024, 16, 2, 8);

        let error = deserialize_row_with_limits(&mut Cursor::new(encoded), 0, limits).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn rdf_literal_encoding_uses_independent_option_flags() {
        let value = Value::RdfLiteral {
            lexical: "x".into(),
            language: Some("EN".into()),
            datatype: Some("urn:Datatype".into()),
        };
        let mut encoded = Vec::new();

        serialize_value(&value, &mut encoded).unwrap();

        let mut expected = vec![TAG_RDF_LITERAL];
        expected.extend_from_slice(&1_u64.to_le_bytes());
        expected.extend_from_slice(b"x");
        expected.push(1);
        expected.extend_from_slice(&2_u64.to_le_bytes());
        expected.extend_from_slice(b"EN");
        expected.push(1);
        expected.extend_from_slice(&12_u64.to_le_bytes());
        expected.extend_from_slice(b"urn:Datatype");
        assert_eq!(encoded, expected);
    }

    #[test]
    fn rdf_literal_roundtrip_preserves_empty_unicode_nul_case_and_both_options() {
        for value in [
            Value::RdfLiteral {
                lexical: "".into(),
                language: None,
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "Grüße\0世界".into(),
                language: Some("EN-gb".into()),
                datatype: None,
            },
            Value::RdfLiteral {
                lexical: "CaseSensitive".into(),
                language: None,
                datatype: Some("urn:TYPE".into()),
            },
            Value::RdfLiteral {
                lexical: "both".into(),
                language: Some("MiXeD".into()),
                datatype: Some("urn:Both".into()),
            },
        ] {
            assert_eq!(roundtrip_value(value.clone()), value);
        }
    }

    #[test]
    fn deserialize_value_rejects_non_boolean_bool_payload() {
        let error = deserialize_value(&mut Cursor::new([TAG_BOOL, 2])).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_rdf_literal_rejects_non_boolean_option_flag() {
        let mut encoded = vec![TAG_RDF_LITERAL];
        encoded.extend_from_slice(&0_u64.to_le_bytes());
        encoded.push(2);

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn exact_time_roundtrip_preserves_minimum_offset() {
        let time = grafeo_common::types::Time::from_nanos(42)
            .expect("test time is valid")
            .with_offset(i32::MIN);

        assert_eq!(roundtrip_value(Value::Time(time)), Value::Time(time));
    }

    #[test]
    fn legacy_time_tag_still_decodes_the_old_none_sentinel() {
        let mut encoded = vec![TAG_TIME];
        encoded.extend_from_slice(&42_u64.to_le_bytes());
        encoded.extend_from_slice(&i32::MIN.to_le_bytes());

        let decoded = deserialize_value(&mut Cursor::new(encoded)).unwrap();

        assert_eq!(
            decoded,
            Value::Time(grafeo_common::types::Time::from_nanos(42).expect("test time is valid"))
        );
    }

    fn append_counter_entry(encoded: &mut Vec<u8>, key: &str, value: u64) {
        encoded.extend_from_slice(&(key.len() as u64).to_le_bytes());
        encoded.extend_from_slice(key.as_bytes());
        encoded.extend_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn deserialize_gcounter_rejects_duplicate_keys() {
        let mut encoded = vec![TAG_GCOUNTER];
        encoded.extend_from_slice(&2_u64.to_le_bytes());
        append_counter_entry(&mut encoded, "replica", 1);
        append_counter_entry(&mut encoded, "replica", 2);

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_pncounter_rejects_duplicate_keys_but_accepts_legacy_order() {
        let mut encoded = vec![TAG_PNCOUNTER];
        encoded.extend_from_slice(&2_u64.to_le_bytes());
        append_counter_entry(&mut encoded, "replica-b", 2);
        append_counter_entry(&mut encoded, "replica-b", 1);
        encoded.extend_from_slice(&0_u64.to_le_bytes());

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        let mut legacy_order = vec![TAG_PNCOUNTER];
        legacy_order.extend_from_slice(&2_u64.to_le_bytes());
        append_counter_entry(&mut legacy_order, "replica-b", 2);
        append_counter_entry(&mut legacy_order, "replica-a", 1);
        legacy_order.extend_from_slice(&0_u64.to_le_bytes());
        assert!(deserialize_value(&mut Cursor::new(legacy_order)).is_ok());
    }

    #[test]
    fn deserialize_value_rejects_nesting_deeper_than_128() {
        let mut encoded = Vec::new();
        for _ in 0..129 {
            encoded.push(TAG_LIST);
            encoded.extend_from_slice(&1_u64.to_le_bytes());
        }
        encoded.push(TAG_NULL);

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_row_enforces_one_cumulative_allocation_budget() {
        const FIRST_VALUE_BYTES: usize = 34 * 1024 * 1024;
        const SECOND_VALUE_BYTES: usize = 31 * 1024 * 1024;
        let mut encoded = Vec::with_capacity(8 + 9 + FIRST_VALUE_BYTES + 9);
        encoded.extend_from_slice(&2_u64.to_le_bytes());
        encoded.push(TAG_BYTES);
        encoded.extend_from_slice(&(FIRST_VALUE_BYTES as u64).to_le_bytes());
        encoded.resize(encoded.len() + FIRST_VALUE_BYTES, 0);
        encoded.push(TAG_BYTES);
        encoded.extend_from_slice(&(SECOND_VALUE_BYTES as u64).to_le_bytes());

        let error =
            deserialize_row_with_limits(&mut Cursor::new(encoded), 2, CodecLimits::default())
                .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn float_and_vector_roundtrip_preserve_bit_patterns() {
        let float_bits = 0x7ff8_0000_0000_0042_u64;
        let vector_bits = [0x8000_0000_u32, 0x7fc0_0042, 0x7f80_0000];

        let float = roundtrip_value(Value::Float64(f64::from_bits(float_bits)));
        assert!(matches!(float, Value::Float64(value) if value.to_bits() == float_bits));

        let vector = roundtrip_value(Value::Vector(Arc::from(vector_bits.map(f32::from_bits))));
        let Value::Vector(vector) = vector else {
            panic!("vector roundtrip changed the value variant");
        };
        assert_eq!(
            vector
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vector_bits
        );
    }

    #[test]
    fn deserialize_map_rejects_duplicate_or_non_increasing_keys() {
        for keys in [["same", "same"], ["later", "earlier"]] {
            let mut encoded = vec![TAG_MAP];
            encoded.extend_from_slice(&2_u64.to_le_bytes());
            for key in keys {
                encoded.extend_from_slice(&(key.len() as u64).to_le_bytes());
                encoded.extend_from_slice(key.as_bytes());
                encoded.push(TAG_NULL);
            }

            let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn default_value_codec_accepts_value_above_conservative_byte_limit() {
        let value = Value::Bytes(Arc::from(vec![0_u8; 67_108_865]));
        let mut encoded = Vec::new();

        serialize_value(&value, &mut encoded).unwrap();
        let decoded = deserialize_value(&mut Cursor::new(encoded)).unwrap();

        assert_eq!(decoded, value);
    }

    #[test]
    fn default_value_codec_accepts_item_count_above_conservative_limit() {
        let value = Value::List(Arc::from(vec![Value::Null; 1_000_001]));
        let mut encoded = Vec::new();

        serialize_value(&value, &mut encoded).unwrap();
        let decoded = deserialize_value(&mut Cursor::new(encoded)).unwrap();

        assert_eq!(decoded, value);
    }

    #[test]
    fn default_row_codec_accepts_column_count_above_conservative_limit() {
        let row = vec![Value::Null; 65_537];
        let mut encoded = Vec::new();

        serialize_row(&row, &mut encoded).unwrap();
        let decoded = deserialize_row(&mut Cursor::new(encoded), row.len()).unwrap();

        assert_eq!(decoded, row);
    }

    #[test]
    fn serialize_value_rejects_nesting_deeper_than_128() {
        let mut value = Value::Null;
        for _ in 0..129 {
            value = Value::List(Arc::from([value]));
        }

        let mut encoded = Vec::new();
        let error = serialize_value(&value, &mut encoded).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encoded.is_empty());
    }

    #[test]
    fn deserialize_value_enforces_cumulative_nested_item_budget() {
        let mut encoded = vec![TAG_LIST];
        encoded.extend_from_slice(&600_000_u64.to_le_bytes());
        encoded.push(TAG_LIST);
        encoded.extend_from_slice(&600_000_u64.to_le_bytes());

        let error =
            deserialize_value_with_limits(&mut Cursor::new(encoded), CodecLimits::default())
                .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn serialize_row_validates_every_column_before_writing_header() {
        let mut invalid = Value::Null;
        for _ in 0..129 {
            invalid = Value::List(Arc::from([invalid]));
        }
        let row = [Value::Int64(1), invalid];
        let mut encoded = Vec::new();

        let error =
            serialize_row_with_limits(&row, &mut encoded, CodecLimits::default()).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encoded.is_empty());
    }

    #[test]
    fn serialize_row_enforces_one_cumulative_item_budget_before_writing() {
        let row = [
            Value::List(Arc::from(vec![Value::Null; 600_000])),
            Value::List(Arc::from(vec![Value::Null; 400_000])),
        ];
        let mut encoded = Vec::new();

        let error =
            serialize_row_with_limits(&row, &mut encoded, CodecLimits::default()).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encoded.is_empty());
    }

    #[test]
    fn explicit_item_and_column_limits_reject_before_writing() {
        let list = Value::List(Arc::from([Value::Null, Value::Null, Value::Null]));
        let mut encoded = Vec::new();
        let item_limits = CodecLimits::new(1024, 2, 4, 8);

        let item_error = serialize_value_with_limits(&list, &mut encoded, item_limits).unwrap_err();
        assert_eq!(item_error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encoded.is_empty());

        let row = [Value::Null, Value::Null, Value::Null];
        let column_limits = CodecLimits::new(1024, 16, 2, 8);
        let column_error =
            serialize_row_with_limits(&row, &mut encoded, column_limits).unwrap_err();
        assert_eq!(column_error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encoded.is_empty());
    }

    #[test]
    fn explicit_byte_limit_rejects_before_writing_and_can_be_raised() {
        let value = Value::Bytes(Arc::from([1_u8, 2, 3, 4, 5]));
        let mut encoded = Vec::new();
        let small = CodecLimits::new(4, 16, 4, 8);

        let error = serialize_value_with_limits(&value, &mut encoded, small).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(encoded.is_empty());

        let raised = CodecLimits::new(5, 16, 4, 8);
        serialize_value_with_limits(&value, &mut encoded, raised).unwrap();
        let decoded = deserialize_value_with_limits(&mut Cursor::new(encoded), raised).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn codec_limit_accessors_report_the_clamped_operation_budget() {
        let limits = CodecLimits::new(123, 45, 6, 7);

        assert_eq!(limits.max_bytes(), 123);
        assert_eq!(limits.max_items(), 45);
    }

    #[test]
    fn caller_limits_cannot_exceed_the_platform_allocation_format_maximum() {
        let mut encoded = vec![TAG_BYTES];
        encoded.extend_from_slice(&u64::MAX.to_le_bytes());
        let limits = CodecLimits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX);

        let error = deserialize_value_with_limits(&mut Cursor::new(encoded), limits).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_exact_time_rejects_non_boolean_presence_flag() {
        let mut encoded = vec![TAG_TIME_EXACT];
        encoded.extend_from_slice(&42_u64.to_le_bytes());
        encoded.push(2);

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn deserialize_rdf_literal_rejects_truncated_option_payload() {
        let mut encoded = vec![TAG_RDF_LITERAL];
        encoded.extend_from_slice(&1_u64.to_le_bytes());
        encoded.push(b'x');
        encoded.push(1);
        encoded.extend_from_slice(&2_u64.to_le_bytes());
        encoded.push(b'e');

        let error = deserialize_value(&mut Cursor::new(encoded)).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
