//! Fixed-width resume coordinates for the retained native change feed.

use super::{EpochId, StoreId};
use crate::utils::error::{Result, StorageError};

/// One store-wide feed. Model tags are 1 (LPG), 2 (RDF), and 3 (mixed).
/// Identifier zero is the only currently defined feed identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FeedId {
    /// Native model tag, independent of graph paths and incarnations.
    pub model: u8,
    /// Store-wide feed identifier; currently always zero.
    pub identifier: u64,
}

impl FeedId {
    /// Creates a supported store-wide feed identity.
    ///
    /// # Errors
    /// Rejects unknown model tags and nonzero feed identifiers.
    pub fn new(model: u8, identifier: u64) -> Result<Self> {
        let feed = Self { model, identifier };
        feed.validate()?;
        Ok(feed)
    }

    fn validate(self) -> Result<()> {
        if !matches!(self.model, 1..=3) || self.identifier != 0 {
            return Err(StorageError::CursorInvalid.into());
        }
        Ok(())
    }
}

/// Exclusive resume position in one retained feed generation.
///
/// The unkeyed digest detects altered canonical fields; it is not authorization.
/// Database admission additionally validates store, generation, retained range,
/// and the event's epoch. Use [`Self::to_bytes`] for portable cursor transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DurableCursor {
    /// Stable logical store identity.
    pub store_id: StoreId,
    /// Store-wide native feed identity.
    pub feed: FeedId,
    /// Retained feed generation.
    pub generation: u64,
    /// Last consumed sequence, or the position immediately before the floor.
    pub sequence: u64,
    /// Commit epoch at the consumed position; zero for an initial position.
    pub epoch: EpochId,
    /// Domain-separated BLAKE3 digest of all preceding canonical fields.
    pub digest: [u8; 32],
}

impl DurableCursor {
    /// Exact length of the current canonical cursor encoding.
    pub const LEN: usize = 97;

    /// Creates a canonical cursor. This does not assert that a store retains it.
    ///
    /// # Errors
    /// Rejects unsupported feeds and reserved generation/sequence/epoch values.
    pub fn new(
        store_id: StoreId,
        feed: FeedId,
        generation: u64,
        sequence: u64,
        epoch: EpochId,
    ) -> Result<Self> {
        let mut cursor = Self {
            store_id,
            feed,
            generation,
            sequence,
            epoch,
            digest: [0; 32],
        };
        cursor.validate_fields()?;
        cursor.digest = cursor.computed_digest();
        Ok(cursor)
    }

    fn validate_fields(&self) -> Result<()> {
        self.feed.validate()?;
        if matches!(self.generation, 0 | u64::MAX)
            || self.sequence == u64::MAX
            || self.epoch == EpochId::PENDING
        {
            return Err(StorageError::CursorInvalid.into());
        }
        Ok(())
    }

    fn computed_digest(&self) -> [u8; 32] {
        use blake3::hazmat::HasherExt;

        // Reuse only the fixed domain context. Every cursor's coordinates still
        // pass through the same BLAKE3 derive-key mode and canonical encoding.
        static CONTEXT_KEY: std::sync::LazyLock<blake3::hazmat::ContextKey> =
            std::sync::LazyLock::new(|| {
                blake3::hazmat::hash_derive_key_context("grafeo/durable-cursor/v1")
            });
        let bytes = self.to_bytes();
        let mut hasher = blake3::Hasher::new_from_context_key(&CONTEXT_KEY);
        hasher.update(&bytes[..65]);
        *hasher.finalize().as_bytes()
    }

    /// Validates the supported fields and their exact digest.
    ///
    /// # Errors
    /// Returns a structured invalid-cursor error for malformed authority.
    pub fn validate(&self) -> Result<()> {
        self.validate_fields()?;
        if self.digest != self.computed_digest() {
            return Err(StorageError::CursorInvalid.into());
        }
        Ok(())
    }

    /// Encodes fixed-width fields with little-endian integer coordinates.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::LEN] {
        let mut bytes = [0; Self::LEN];
        bytes[..32].copy_from_slice(self.store_id.as_bytes());
        bytes[32] = self.feed.model;
        bytes[33..41].copy_from_slice(&self.feed.identifier.to_le_bytes());
        bytes[41..49].copy_from_slice(&self.generation.to_le_bytes());
        bytes[49..57].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[57..65].copy_from_slice(&self.epoch.as_u64().to_le_bytes());
        bytes[65..].copy_from_slice(&self.digest);
        bytes
    }

    /// Decodes only the exact current encoding and validates its digest.
    ///
    /// # Errors
    /// Rejects truncated, extended, unsupported, reserved, or altered cursors.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        fn field<const N: usize>(bytes: &[u8], start: usize) -> Result<[u8; N]> {
            bytes
                .get(start..start + N)
                .and_then(|field| field.try_into().ok())
                .ok_or_else(|| StorageError::CursorInvalid.into())
        }
        if bytes.len() != Self::LEN {
            return Err(StorageError::CursorInvalid.into());
        }
        let cursor = Self {
            store_id: StoreId::from_bytes(field(bytes, 0)?)
                .map_err(|_| StorageError::CursorInvalid)?,
            feed: FeedId {
                model: bytes[32],
                identifier: u64::from_le_bytes(field(bytes, 33)?),
            },
            generation: u64::from_le_bytes(field(bytes, 41)?),
            sequence: u64::from_le_bytes(field(bytes, 49)?),
            epoch: EpochId::new(u64::from_le_bytes(field(bytes, 57)?)),
            digest: field(bytes, 65)?,
        };
        cursor.validate()?;
        Ok(cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::error::ErrorCode;

    fn cursor() -> DurableCursor {
        DurableCursor::new(
            StoreId::from_bytes([0x42; 32]).unwrap(),
            FeedId::new(3, 0).unwrap(),
            7,
            0x0102_0304_0506_0708,
            EpochId::new(19),
        )
        .unwrap()
    }

    #[test]
    fn fixed_cursor_layout_and_every_byte_are_authenticated() {
        let cursor = cursor();
        let bytes = cursor.to_bytes();
        assert_eq!(
            cursor.digest,
            blake3::derive_key("grafeo/durable-cursor/v1", &bytes[..65]),
            "context reuse must preserve the original canonical digest"
        );
        assert_eq!(DurableCursor::from_bytes(&bytes).unwrap(), cursor);
        assert_eq!(&bytes[49..57], &[8, 7, 6, 5, 4, 3, 2, 1]);
        for index in 0..bytes.len() {
            let mut altered = bytes;
            altered[index] ^= 1;
            assert_eq!(
                DurableCursor::from_bytes(&altered)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorInvalid,
                "byte {index}"
            );
        }
        for end in 0..bytes.len() {
            assert!(DurableCursor::from_bytes(&bytes[..end]).is_err());
        }
        let mut extended = bytes.to_vec();
        extended.push(0);
        assert!(DurableCursor::from_bytes(&extended).is_err());
    }

    #[test]
    fn supported_fields_are_checked_even_before_digest_generation() {
        let original = cursor();
        for (feed, generation, sequence, epoch) in [
            (
                FeedId {
                    model: 0,
                    identifier: 0,
                },
                1,
                0,
                EpochId::INITIAL,
            ),
            (
                FeedId {
                    model: 4,
                    identifier: 0,
                },
                1,
                0,
                EpochId::INITIAL,
            ),
            (
                FeedId {
                    model: 1,
                    identifier: 1,
                },
                1,
                0,
                EpochId::INITIAL,
            ),
            (original.feed, 0, 0, EpochId::INITIAL),
            (original.feed, u64::MAX, 0, EpochId::INITIAL),
            (original.feed, 1, u64::MAX, EpochId::INITIAL),
            (original.feed, 1, 0, EpochId::PENDING),
        ] {
            assert_eq!(
                DurableCursor::new(original.store_id, feed, generation, sequence, epoch)
                    .unwrap_err()
                    .error_code(),
                ErrorCode::CursorInvalid
            );
        }
    }

    #[test]
    fn cursor_errors_keep_distinct_public_transport_codes() {
        for (kind, code) in [
            (StorageError::CursorInvalid, "GRAFEO-S004"),
            (StorageError::CursorForeign, "GRAFEO-S005"),
            (StorageError::CursorEvicted, "GRAFEO-S006"),
        ] {
            let error: crate::utils::error::Error = kind.into();
            assert_eq!(error.error_code().as_str(), code);
            assert!(error.to_string().starts_with(code));
        }
    }
}
