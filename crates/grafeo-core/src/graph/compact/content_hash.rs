//! Physical content hashing for the temporal cold base.
//!
//! [`crate::graph::compact::content_hash::physical_block_hash`] is grafeo's OWN Merkle-leaf hash over a serialized
//! columnar block — used for physical dedup (unchanged blocks across epochs share
//! one [`grafeo_common::types::ContentId`]) and scrub integrity. This is a **physical** identity,
//! deliberately distinct from the *semantic* content-ids an application supplies
//! (which grafeo mirrors opaquely and never recomputes). Evidence verification
//! remains the source-of-truth's CT tree; this hash only certifies grafeo's local
//! layout.

use grafeo_common::types::ContentId;

/// Computes the BLAKE3 physical content hash of a block's bytes.
///
/// Deterministic across architectures (same bytes ⇒ same [`ContentId`]) — the
/// property that makes cross-time block dedup and the projection determinism
/// contract hold.
#[must_use]
pub fn physical_block_hash(bytes: &[u8]) -> ContentId {
    ContentId::from_bytes(*blake3::hash(bytes).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        assert_eq!(
            physical_block_hash(b"some block"),
            physical_block_hash(b"some block")
        );
    }

    #[test]
    fn test_distinct_inputs_distinct_hashes() {
        assert_ne!(
            physical_block_hash(b"block-a"),
            physical_block_hash(b"block-b")
        );
    }

    #[test]
    fn test_known_vector_abc() {
        // BLAKE3("abc") official test vector (first 32 bytes).
        assert_eq!(
            format!("{}", physical_block_hash(b"abc")),
            "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
        );
    }

    #[test]
    fn test_known_vector_empty() {
        // BLAKE3("") official test vector.
        assert_eq!(
            format!("{}", physical_block_hash(b"")),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }
}
