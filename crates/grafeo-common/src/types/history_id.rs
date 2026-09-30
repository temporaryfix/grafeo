//! Stable identities and provenance bounds for durable graph history.
//!
//! These types deliberately contain no storage policy.  The engine owns store
//! identity generation, named-graph incarnation allocation, and statement
//! hashing; this module only gives those values a portable, non-truncating
//! representation shared by core, persistence, and bindings.

use core::fmt;
use core::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::EpochId;

/// Error returned when a stable 256-bit identifier is malformed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{kind} must be exactly 64 lowercase or uppercase hexadecimal digits")]
pub struct ParseStableIdError {
    kind: &'static str,
}

impl ParseStableIdError {
    const fn new(kind: &'static str) -> Self {
        Self { kind }
    }
}

/// Portable identity of one logical Grafeo store.
///
/// A store keeps this value across checkpoint, snapshot transfer, and reopen.
/// Cloning bytes into a distinct logical store must allocate a new identity;
/// restoring the same store preserves it.  The all-zero value is reserved and
/// rejected so an omitted legacy field cannot silently become an identity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StoreId([u8; 32]);

impl StoreId {
    /// Number of bytes in a store identity.
    pub const LEN: usize = 32;

    /// Constructs a non-zero store identity from its portable bytes.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidStoreId`] for the reserved all-zero value.
    pub const fn from_bytes(bytes: [u8; Self::LEN]) -> Result<Self, InvalidStoreId> {
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] != 0 {
                return Ok(Self(bytes));
            }
            index += 1;
        }
        Err(InvalidStoreId)
    }

    /// Returns the exact portable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::LEN] {
        &self.0
    }

    /// Consumes the identity into its portable bytes.
    #[must_use]
    pub const fn into_bytes(self) -> [u8; Self::LEN] {
        self.0
    }

    /// Generates a new logical-store identity from the operating system's
    /// cryptographically secure random source.
    ///
    /// # Errors
    ///
    /// Returns [`StoreIdGenerationError`] when the system entropy source is
    /// unavailable. The reserved all-zero output is discarded and retried.
    pub fn generate() -> Result<Self, StoreIdGenerationError> {
        loop {
            let mut bytes = [0_u8; Self::LEN];
            getrandom::fill(&mut bytes).map_err(StoreIdGenerationError)?;
            if let Ok(store_id) = Self::from_bytes(bytes) {
                return Ok(store_id);
            }
        }
    }
}

/// The reserved all-zero store identity was supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("store identity must not be all zero")]
pub struct InvalidStoreId;

/// The operating system could not generate a logical-store identity.
#[derive(Debug, thiserror::Error)]
#[error("failed to generate store identity from system entropy: {0}")]
pub struct StoreIdGenerationError(#[source] getrandom::Error);

impl TryFrom<[u8; StoreId::LEN]> for StoreId {
    type Error = InvalidStoreId;

    fn try_from(bytes: [u8; StoreId::LEN]) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

impl Serialize for StoreId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StoreId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = <[u8; StoreId::LEN]>::deserialize(deserializer)?;
        Self::from_bytes(bytes).map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for StoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StoreId({self})")
    }
}

impl fmt::Display for StoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(&self.0, f)
    }
}

impl FromStr for StoreId {
    type Err = ParseStableIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = parse_256_bit_hex(value, "store identity")?;
        Self::from_bytes(bytes).map_err(|_| ParseStableIdError::new("store identity"))
    }
}

/// Identity of one native graph lifetime, qualified by StoreId and graph model.
///
/// Incarnation `0` is permanently reserved for the default graph.  Named graph
/// incarnations start at `1` and are never reused, including after DROP and
/// CREATE of the same LPG path or RDF graph IRI. LPG and RDF have independent
/// native allocators; equal integers across the two models are not equal owners.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[repr(transparent)]
pub struct GraphIncarnationId(u64);

impl GraphIncarnationId {
    /// Permanent incarnation of a native model's default graph.
    pub const DEFAULT_GRAPH: Self = Self(0);
    /// First allocatable named-graph incarnation.
    pub const FIRST_NAMED: Self = Self(1);

    /// Constructs an incarnation identifier from its durable integer.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the durable integer representation.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether this is the reserved default-graph incarnation.
    #[must_use]
    pub const fn is_default_graph(self) -> bool {
        self.0 == Self::DEFAULT_GRAPH.0
    }

    /// Returns the next incarnation, or `None` at exhaustion.
    #[must_use]
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl fmt::Display for GraphIncarnationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stable, non-truncated identity of one RDF statement in one graph lifetime.
///
/// The core RDF history layer computes this with domain-separated BLAKE3 over
/// the domain version, [`StoreId`], lossless RDF terms, graph name, and
/// [`GraphIncarnationId`]. Keeping the full 256 bits avoids the collision
/// ambiguity of the legacy CDC `u64` hash and prevents cross-store aliasing.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct StatementHandle([u8; 32]);

impl StatementHandle {
    /// Number of bytes in a statement handle.
    pub const LEN: usize = 32;

    /// Constructs a handle from the complete digest.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; Self::LEN]) -> Self {
        Self(bytes)
    }

    /// Returns the complete digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::LEN] {
        &self.0
    }

    /// Consumes the handle into its complete digest bytes.
    #[must_use]
    pub const fn into_bytes(self) -> [u8; Self::LEN] {
        self.0
    }
}

impl fmt::Debug for StatementHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StatementHandle({self})")
    }
}

impl fmt::Display for StatementHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(&self.0, f)
    }
}

impl FromStr for StatementHandle {
    type Err = ParseStableIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_256_bit_hex(value, "statement handle").map(Self)
    }
}

/// Provenance boundary for persisted RDF temporal history.
///
/// Legacy formats stored only the graph state visible when they were written.
/// Loading one must use [`LegacyCurrentState`](Self::LegacyCurrentState), not
/// manufacture assertion epochs or claim an audit trail that never existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HistoryCompleteness {
    /// Every graph lifecycle and statement transition is represented.
    Complete,
    /// A legacy source provided current state at `observed_at`, but no reliable
    /// transition history before that boundary.
    LegacyCurrentState {
        /// First epoch from which newly committed transitions are authoritative.
        observed_at: EpochId,
        /// Version number of the legacy source format.
        source_version: u16,
    },
}

impl HistoryCompleteness {
    /// Whether the complete history is available from the beginning of time.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Complete)
    }

    /// Earliest epoch from which transition history is authoritative.
    ///
    /// `None` means the complete history is available.
    #[must_use]
    pub const fn authoritative_from(self) -> Option<EpochId> {
        match self {
            Self::Complete => None,
            Self::LegacyCurrentState { observed_at, .. } => Some(observed_at),
        }
    }

    /// Whether transitions strictly after `from` are covered.
    #[must_use]
    pub const fn covers_after(self, from: EpochId) -> bool {
        match self.authoritative_from() {
            None => true,
            Some(boundary) => from.as_u64() >= boundary.as_u64(),
        }
    }
}

fn write_hex(bytes: &[u8; 32], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        f.write_str(
            core::str::from_utf8(&[DIGITS[(byte >> 4) as usize], DIGITS[(byte & 0x0f) as usize]])
                .expect("hex digits are UTF-8"),
        )?;
    }
    Ok(())
}

fn parse_256_bit_hex(value: &str, kind: &'static str) -> Result<[u8; 32], ParseStableIdError> {
    if value.len() != 64 {
        return Err(ParseStableIdError::new(kind));
    }
    let mut bytes = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or_else(|| ParseStableIdError::new(kind))?;
        let low = hex_nibble(pair[1]).ok_or_else(|| ParseStableIdError::new(kind))?;
        bytes[index] = (high << 4) | low;
    }
    Ok(bytes)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_round_trip_without_truncation() {
        let mut bytes = [0u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::try_from(index).expect("32-byte identity index fits in u8");
        }
        let store = StoreId::from_bytes(bytes).unwrap();
        assert_eq!(store.to_string().parse::<StoreId>().unwrap(), store);

        let handle = StatementHandle::from_bytes(bytes);
        assert_eq!(
            handle.to_string().parse::<StatementHandle>().unwrap(),
            handle
        );
        assert_eq!(handle.to_string().len(), 64);
    }

    #[test]
    fn store_identity_rejects_legacy_zero_sentinel() {
        assert_eq!(StoreId::from_bytes([0; 32]), Err(InvalidStoreId));
        assert!("00".repeat(32).parse::<StoreId>().is_err());
    }

    #[test]
    fn named_incarnations_never_alias_default() {
        assert!(GraphIncarnationId::DEFAULT_GRAPH.is_default_graph());
        assert!(!GraphIncarnationId::FIRST_NAMED.is_default_graph());
        assert_eq!(
            GraphIncarnationId::FIRST_NAMED.checked_next(),
            Some(GraphIncarnationId::new(2))
        );
    }

    #[test]
    fn legacy_history_exposes_its_authoritative_boundary() {
        let completeness = HistoryCompleteness::LegacyCurrentState {
            observed_at: EpochId::new(7),
            source_version: 4,
        };
        assert!(!completeness.is_complete());
        assert!(!completeness.covers_after(EpochId::new(6)));
        assert!(completeness.covers_after(EpochId::new(7)));
    }
}
