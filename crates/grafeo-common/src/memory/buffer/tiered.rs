//! Storage-tier state for data structures served from RAM or mapped disk.
//!
//! [`Section`](crate::storage::section::Section) provides the persistence and
//! tier-transition protocol. [`MemoryConsumer`](super::MemoryConsumer) reports
//! managed memory use and participates in eviction through the buffer manager.

/// The current storage tier of a data structure.
///
/// Sections and memory consumers expose the canonical lifecycle and accounting
/// interfaces alongside this state:
///
/// ```
/// use grafeo_common::memory::{MemoryConsumer, StorageTier};
/// use grafeo_common::storage::section::Section;
///
/// fn inspect(section: &dyn Section, consumer: &dyn MemoryConsumer) -> (u8, usize) {
///     (section.version(), consumer.memory_usage())
/// }
///
/// assert!(StorageTier::InMemory.is_in_memory());
/// assert!(StorageTier::OnDisk.is_on_disk());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StorageTier {
    /// Fully in RAM. Fastest access for both reads and writes.
    InMemory,
    /// On disk, accessed via mmap. The OS page cache provides warm reads.
    /// Mutations go through a WAL overlay.
    OnDisk,
    /// Not yet initialized (structure exists but has no data).
    Uninitialized,
}

impl StorageTier {
    /// Returns `true` if data is fully in RAM.
    #[must_use]
    pub fn is_in_memory(self) -> bool {
        self == Self::InMemory
    }

    /// Returns `true` if data is served from disk (mmap).
    #[must_use]
    pub fn is_on_disk(self) -> bool {
        self == Self::OnDisk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_predicates() {
        assert!(StorageTier::InMemory.is_in_memory());
        assert!(!StorageTier::InMemory.is_on_disk());

        assert!(StorageTier::OnDisk.is_on_disk());
        assert!(!StorageTier::OnDisk.is_in_memory());

        assert!(!StorageTier::Uninitialized.is_in_memory());
        assert!(!StorageTier::Uninitialized.is_on_disk());
    }

    #[test]
    fn test_storage_tier_equality() {
        assert_eq!(StorageTier::InMemory, StorageTier::InMemory);
        assert_ne!(StorageTier::InMemory, StorageTier::OnDisk);
        assert_ne!(StorageTier::OnDisk, StorageTier::Uninitialized);
    }
}
