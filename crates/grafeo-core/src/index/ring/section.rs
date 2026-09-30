//! Ring Index section for `.grafeo` container persistence.
//!
//! Serializes and deserializes the [`super::TripleRing`] via the
//! [`Section`] trait, enabling the Ring to survive database restarts
//! without rebuilding from triples.
//!
//! The section has one canonical `GRFR` packed format with a CRC32 trailer.
//! Reads are mmap-friendly via refcounted `Bytes` slices and reject every
//! other grammar rather than carrying predecessor readers in the storage kernel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::memory::buffer::SpillError;
use grafeo_common::storage::page_fetcher::PageFetcher;
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::graph::rdf::RdfStore;

/// On-disk version of the canonical packed Ring section.
const RING_SECTION_VERSION: u8 = 2;

/// First four bytes of the canonical Ring envelope.
#[cfg(test)]
const RING_MAGIC: &[u8; 4] = b"GRFR";

/// Section implementation for the RDF Ring Index.
///
/// Wraps an `Arc<RdfStore>` and serializes/deserializes its packed Ring.
pub struct RdfRingSection {
    store: Arc<RdfStore>,
    dirty: AtomicBool,
}

impl RdfRingSection {
    /// Creates a new Ring section backed by the given RDF store.
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            dirty: AtomicBool::new(false),
        }
    }

    /// Marks the section as dirty (Ring was rebuilt or invalidated).
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

impl Section for RdfRingSection {
    fn section_type(&self) -> SectionType {
        SectionType::RdfRing
    }

    fn version(&self) -> u8 {
        RING_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        match self.store.ring() {
            Some(ring) => super::serialize_triple_ring(&ring)
                .map_err(|error| Error::Serialization(error.to_string())),
            None => Ok(Vec::new()),
        }
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let ring = super::deserialize_triple_ring(bytes::Bytes::copy_from_slice(data))
            .map_err(|error| Error::Serialization(error.to_string()))?;
        self.store.set_ring(ring);
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.ring().map_or(0, |r| r.size_bytes())
    }

    /// Swaps the ring backing to a `Bytes` view sourced from `fetcher`.
    ///
    /// After the section serializes to a spill file
    /// and the file is mmap'd, the buffer manager calls this with a
    /// `MmapPageFetcher`. We copy the section into one owning `Bytes`; packed
    /// wavelet level storage then retains refcounted slices of that buffer.
    /// The term dictionary and permutation lookup structures are rebuilt and
    /// own their query-hot state. A future `PageFetcher::owned_bytes` override
    /// on the mmap implementation could eliminate the initial section copy.
    ///
    fn swap_to_mmap(&self, fetcher: Arc<dyn PageFetcher>) -> std::result::Result<(), SpillError> {
        let len = fetcher.len();
        if len == 0 {
            return Ok(());
        }

        let slice = fetcher
            .fetch(0, len)
            .map_err(|e| SpillError::IoError(e.to_string()))?;
        let data = bytes::Bytes::copy_from_slice(slice);

        let ring = super::deserialize_triple_ring(data)
            .map_err(|error| SpillError::IoError(error.to_string()))?;

        self.store.set_ring(ring);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{Term, Triple};

    fn test_store() -> Arc<RdfStore> {
        let store = Arc::new(RdfStore::new());
        store.bulk_load(vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://ex.org/gus"),
            ),
        ]);
        store
    }

    #[test]
    fn section_type_is_rdf_ring() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        assert_eq!(section.section_type(), SectionType::RdfRing);
        assert_eq!(section.version(), 2);
    }

    #[test]
    fn section_dirty_tracking() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn section_serialize_empty() {
        let store = Arc::new(RdfStore::new());
        let section = RdfRingSection::new(store);
        let bytes = section.serialize().unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn section_roundtrip() {
        let store = test_store();
        let section = RdfRingSection::new(Arc::clone(&store));

        // Serialize
        let bytes = section.serialize().unwrap();
        assert!(!bytes.is_empty());

        // Create a fresh store and deserialize into it
        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfRingSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        // The loaded ring should have the same triple count
        let ring = store2.ring().expect("ring should be loaded");
        assert_eq!(ring.len(), 3);

        // Verify count operations work
        use crate::graph::rdf::TriplePattern;
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(ring.count(&name_pattern), 2);
    }

    #[test]
    fn section_memory_usage() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        assert!(section.memory_usage() > 0);
    }

    /// Writes produce the canonical buffer (starting with `GRFR` magic).
    #[test]
    fn alix_section_serialize_writes_ring_magic() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        let bytes = section.serialize().unwrap();
        assert!(bytes.len() > 4);
        assert_eq!(&bytes[0..4], RING_MAGIC);
    }

    /// Unknown predecessor bytes are rejected without replacing an existing
    /// in-memory Ring.
    #[test]
    fn gus_section_rejects_predecessor_without_mutating_store() {
        let store2 = test_store();
        let before = store2.ring().expect("ring built");
        let mut section2 = RdfRingSection::new(Arc::clone(&store2));
        assert!(section2.deserialize(&[0x01; 64]).is_err());
        assert!(Arc::ptr_eq(&store2.ring().expect("ring retained"), &before));
    }

    #[test]
    fn altered_version_is_rejected_without_mutating_store() {
        let source = test_store();
        let mut bytes = RdfRingSection::new(source).serialize().expect("serialize");
        bytes[4] = 1;

        let target = test_store();
        let before = target.ring().expect("ring built");
        let mut section = RdfRingSection::new(Arc::clone(&target));
        assert!(section.deserialize(&bytes).is_err());
        assert!(Arc::ptr_eq(&target.ring().expect("ring retained"), &before));
    }

    /// Minimal in-memory PageFetcher for testing swap_to_mmap.
    struct MemFetcher(Vec<u8>);

    impl PageFetcher for MemFetcher {
        fn fetch(&self, offset: usize, len: usize) -> std::io::Result<&[u8]> {
            let end = offset
                .checked_add(len)
                .ok_or_else(|| std::io::Error::other("overflow"))?;
            if end > self.0.len() {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
            }
            Ok(&self.0[offset..end])
        }

        fn len(&self) -> usize {
            self.0.len()
        }

        fn advise(
            &self,
            _offset: usize,
            _len: usize,
            _hint: grafeo_common::storage::page_fetcher::AccessHint,
        ) {
        }
    }

    /// `swap_to_mmap` rebuilds the ring from a `Bytes`-backed packed buffer
    /// and queries against the swapped-in ring give correct results.
    #[test]
    fn shosanna_swap_to_mmap_serves_queries_from_bytes() {
        let original = test_store();
        let packed_bytes = {
            let section = RdfRingSection::new(Arc::clone(&original));
            section.serialize().unwrap()
        };

        // Fresh empty store; section starts with no ring.
        let store = Arc::new(RdfStore::new());
        assert!(store.ring().is_none());

        let section = RdfRingSection::new(Arc::clone(&store));
        let fetcher: Arc<dyn PageFetcher> = Arc::new(MemFetcher(packed_bytes));
        section.swap_to_mmap(fetcher).expect("swap_to_mmap");

        // Ring is now populated from the fetcher bytes.
        let ring = store.ring().expect("ring loaded via swap_to_mmap");
        assert_eq!(ring.len(), 3);

        // Query semantics still work against the post-swap ring.
        use crate::graph::rdf::TriplePattern;
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(ring.count(&name_pattern), 2);
    }

    /// Empty fetcher (zero-length section) is a no-op, not an error.
    #[test]
    fn butch_swap_to_mmap_empty_fetcher_is_noop() {
        let store = Arc::new(RdfStore::new());
        let section = RdfRingSection::new(Arc::clone(&store));
        let fetcher: Arc<dyn PageFetcher> = Arc::new(MemFetcher(Vec::new()));
        section.swap_to_mmap(fetcher).expect("empty swap is ok");
        assert!(store.ring().is_none());
    }

    #[test]
    fn swap_to_mmap_rejects_altered_version_without_mutation() {
        let source = test_store();
        let mut bytes = RdfRingSection::new(source).serialize().expect("serialize");
        bytes[4] = 1;

        let target = test_store();
        let before = target.ring().expect("ring built");
        let section = RdfRingSection::new(Arc::clone(&target));
        let fetcher: Arc<dyn PageFetcher> = Arc::new(MemFetcher(bytes));
        assert!(section.swap_to_mmap(fetcher).is_err());
        assert!(Arc::ptr_eq(&target.ring().expect("ring retained"), &before));
    }
}
