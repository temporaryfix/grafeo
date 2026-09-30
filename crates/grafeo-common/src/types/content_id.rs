//! Content-addressed identity types for the unified temporal-projection model.
//!
//! Grafeo stores supplied content addresses opaquely; producers define their meaning.
//! - [`EntityRef`] is a stable, 16-byte mutable-entity identity.
//! - [`ContentId`] is a 32-byte identity for an immutable version, block or blob.
//! - [`EvidenceRef`] is an opaque semantic reference supplied by an application.
//!
//! [`LocalId`] is the *private*, dense, per-store cache offset; never durable,
//! never synced, regenerated at compaction. [`Interner`] is the bidirectional
//! content-key ↔ `LocalId` dictionary.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::utils::hash::FxHashMap;

/// A stable, content-derived identity for a *mutable entity* across its versions.
///
/// The durable reference in a version-DAG model. Supplied by the producer
/// and treated as opaque; Grafeo never computes it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct EntityRef(pub [u8; 16]);

impl EntityRef {
    /// The invalid/null entity ref (all-0xFF).
    pub const INVALID: Self = Self([u8::MAX; 16]);

    /// Wraps 16 raw bytes of a supplied content-address.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// The raw 16 bytes.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Checks if this is a valid (non-sentinel) entity ref.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.0 != [u8::MAX; 16]
    }
}

impl fmt::Debug for EntityRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EntityRef({self})")
    }
}

impl fmt::Display for EntityRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl From<[u8; 16]> for EntityRef {
    fn from(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

/// A 32-byte content address for an immutable version, columnar block or blob.
/// Carries supplied hashes verbatim; this module does not compute them.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct ContentId(pub [u8; 32]);

impl ContentId {
    /// The invalid/null content id (all-0xFF).
    pub const INVALID: Self = Self([u8::MAX; 32]);

    /// Wraps 32 raw bytes of a supplied content-address.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32 bytes.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Checks if this is a valid (non-sentinel) content id.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.0 != [u8::MAX; 32]
    }
}

impl fmt::Debug for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentId({self})")
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl From<[u8; 32]> for ContentId {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// An opaque application-supplied semantic identity.
/// Grafeo stores it as a durable indexed key without parsing or re-deriving it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct EvidenceRef(pub Box<str>);

impl EvidenceRef {
    /// Wraps a supplied opaque reference string.
    #[inline]
    #[must_use]
    pub fn new(s: impl Into<Box<str>>) -> Self {
        Self(s.into())
    }

    /// The opaque reference string.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for EvidenceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EvidenceRef({:?})", self.0)
    }
}

impl fmt::Display for EvidenceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for EvidenceRef {
    fn from(s: String) -> Self {
        Self(s.into_boxed_str())
    }
}

impl From<&str> for EvidenceRef {
    fn from(s: &str) -> Self {
        Self(Box::from(s))
    }
}

/// A **private**, dense, per-store cache offset — the cache-friendly hot handle
/// for internal structures (CSR, columns, intermediates).
///
/// NOT durable, NOT synced, NOT part of any public/wire contract. Regenerated at
/// compaction; only the content keys it maps to survive. (Mirrors the role of
/// CompactStore's existing compact-id / `EntityHandle`.)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct LocalId(pub u32);

impl LocalId {
    /// The invalid/null local id.
    pub const INVALID: Self = Self(u32::MAX);

    /// Creates a new `LocalId` from a raw u32.
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw u32 value.
    #[inline]
    #[must_use]
    pub const fn as_u32(&self) -> u32 {
        self.0
    }

    /// Checks if this is a valid local id.
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Debug for LocalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "LocalId({})", self.0)
        } else {
            write!(f, "LocalId(INVALID)")
        }
    }
}

impl fmt::Display for LocalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Bidirectional content-key ↔ dense [`LocalId`] dictionary.
///
/// Interns durable content keys (e.g. [`EntityRef`], [`ContentId`]) to dense,
/// private `LocalId`s for the cache-friendly hot path. **Regenerable:** rebuilding
/// from the same key sequence yields the same assignment (insertion order), so the
/// durable keys — not the offsets — are what survive compaction/sync. (Locality-
/// preserving reassignment at compaction is SP2; SP0a uses insertion order.)
#[derive(Clone, Debug, Default)]
pub struct Interner<K: Copy + Eq + std::hash::Hash> {
    forward: FxHashMap<K, LocalId>,
    reverse: Vec<K>,
}

impl<K: Copy + Eq + std::hash::Hash> Interner<K> {
    /// Creates an empty interner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            forward: FxHashMap::default(),
            reverse: Vec::new(),
        }
    }

    /// Interns a key, returning its (possibly newly assigned) `LocalId`. Idempotent.
    pub fn intern(&mut self, key: K) -> LocalId {
        if let Some(&id) = self.forward.get(&key) {
            return id;
        }
        // reason: store size is bounded by addressable entities; u32 offsets suffice.
        #[allow(clippy::cast_possible_truncation)]
        let id = LocalId::new(self.reverse.len() as u32);
        self.reverse.push(key);
        self.forward.insert(key, id);
        id
    }

    /// Returns the `LocalId` for a key if already interned.
    #[must_use]
    pub fn local_of(&self, key: &K) -> Option<LocalId> {
        self.forward.get(key).copied()
    }

    /// Returns the key for a `LocalId`, if in range.
    #[must_use]
    pub fn resolve(&self, id: LocalId) -> Option<K> {
        self.reverse.get(id.0 as usize).copied()
    }

    /// Number of interned keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.reverse.len()
    }

    /// Whether the interner is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reverse.is_empty()
    }

    /// Deterministic iteration in `LocalId` order.
    pub fn iter(&self) -> impl Iterator<Item = (LocalId, K)> + '_ {
        // reason: index fits u32 by construction (intern() bounds it).
        #[allow(clippy::cast_possible_truncation)]
        self.reverse
            .iter()
            .enumerate()
            .map(|(i, &k)| (LocalId::new(i as u32), k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_ref_basic() {
        let r = EntityRef::from_bytes([7u8; 16]);
        assert_eq!(r.as_bytes(), &[7u8; 16]);
        assert!(r.is_valid());
        assert!(!EntityRef::INVALID.is_valid());
    }

    #[test]
    fn test_entity_ref_ord_is_byte_lexicographic() {
        // Deterministic tie-break primitive: ordering is plain byte order.
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        a[0] = 1;
        b[1] = 1; // a > b lexicographically (first byte dominates)
        assert!(EntityRef::from_bytes(a) > EntityRef::from_bytes(b));
    }

    #[test]
    fn test_entity_ref_display_is_hex() {
        let mut bytes = [0u8; 16];
        bytes[0] = 0xab;
        bytes[15] = 0x0c;
        let s = format!("{}", EntityRef::from_bytes(bytes));
        assert!(s.starts_with("ab"));
        assert!(s.ends_with("0c"));
        assert_eq!(s.len(), 32); // 16 bytes * 2 hex chars
    }

    #[test]
    fn test_content_id_basic() {
        let c = ContentId::from_bytes([3u8; 32]);
        assert_eq!(c.as_bytes(), &[3u8; 32]);
        assert!(c.is_valid());
        assert!(!ContentId::INVALID.is_valid());
    }

    #[test]
    fn test_content_id_display_is_hex_64() {
        let s = format!("{}", ContentId::from_bytes([0xffu8; 32]));
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|ch| ch == 'f'));
    }

    #[test]
    fn test_evidence_ref_opaque_string() {
        let r = EvidenceRef::new("ev:track:abc123");
        assert_eq!(r.as_str(), "ev:track:abc123");
        assert_eq!(format!("{r}"), "ev:track:abc123");
    }

    #[test]
    fn test_evidence_ref_ord_is_string_order() {
        assert!(EvidenceRef::new("a") < EvidenceRef::new("b"));
    }

    #[test]
    fn test_local_id_basic() {
        let l = LocalId::new(5);
        assert_eq!(l.as_u32(), 5);
        assert!(l.is_valid());
        assert!(!LocalId::INVALID.is_valid());
    }

    #[test]
    fn test_interner_intern_is_idempotent_and_dense() {
        let mut it: Interner<EntityRef> = Interner::new();
        let a = EntityRef::from_bytes([1u8; 16]);
        let b = EntityRef::from_bytes([2u8; 16]);
        assert_eq!(it.intern(a), LocalId::new(0));
        assert_eq!(it.intern(b), LocalId::new(1));
        assert_eq!(it.intern(a), LocalId::new(0)); // idempotent
        assert_eq!(it.len(), 2);
    }

    #[test]
    fn test_interner_resolve_and_local_of() {
        let mut it: Interner<ContentId> = Interner::new();
        let c = ContentId::from_bytes([9u8; 32]);
        let id = it.intern(c);
        assert_eq!(it.resolve(id), Some(c));
        assert_eq!(it.local_of(&c), Some(id));
        assert_eq!(it.resolve(LocalId::new(999)), None);
    }

    #[test]
    fn test_interner_assignment_is_deterministic_by_insertion_order() {
        // Same key sequence ⇒ same LocalId assignment (the regenerable property).
        let keys = [
            EntityRef::from_bytes([5u8; 16]),
            EntityRef::from_bytes([1u8; 16]),
            EntityRef::from_bytes([9u8; 16]),
        ];
        let mut a: Interner<EntityRef> = Interner::new();
        let mut b: Interner<EntityRef> = Interner::new();
        for k in keys {
            a.intern(k);
        }
        for k in keys {
            b.intern(k);
        }
        let av: Vec<_> = a.iter().collect();
        let bv: Vec<_> = b.iter().collect();
        assert_eq!(av, bv);
        assert_eq!(av[0], (LocalId::new(0), keys[0])); // iteration in LocalId order
    }
}
