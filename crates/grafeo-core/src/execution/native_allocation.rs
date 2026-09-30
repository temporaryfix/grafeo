//! Stable execution-layer allocation failures for version-pinned containers.

#[cfg(test)]
use std::alloc::Layout;

/// Stable classification of a native partition-map reservation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeMapAllocationKind {
    /// The requested table capacity exceeded the addressable layout.
    CapacityOverflow,
    /// The allocator refused an otherwise valid layout.
    AllocatorRefused {
        /// Requested allocation size.
        requested_bytes: usize,
        /// Requested allocation alignment.
        alignment: usize,
    },
}

/// Grafeo-owned error envelope for a native partition-map reservation.
///
/// Production failures retain the exact dependency error as their source,
/// while the public type and classification remain independent of the pinned
/// hash-table implementation. This keeps a future dependency upgrade out of
/// Grafeo's public semver surface.
#[derive(Clone, Debug)]
pub struct NativeMapAllocationError {
    kind: NativeMapAllocationKind,
    source: Option<hashbrown::TryReserveError>,
}

impl NativeMapAllocationError {
    /// Creates a dependency-independent capacity-overflow error.
    ///
    /// Native map operations use the private source-retaining constructor;
    /// this constructor supports propagation of an already classified error.
    #[must_use]
    pub const fn capacity_overflow() -> Self {
        Self {
            kind: NativeMapAllocationKind::CapacityOverflow,
            source: None,
        }
    }

    #[cfg(any(feature = "spill", test))]
    pub(crate) fn from_hashbrown(source: hashbrown::TryReserveError) -> Self {
        let kind = match &source {
            hashbrown::TryReserveError::CapacityOverflow => {
                NativeMapAllocationKind::CapacityOverflow
            }
            hashbrown::TryReserveError::AllocError { layout } => {
                NativeMapAllocationKind::AllocatorRefused {
                    requested_bytes: layout.size(),
                    alignment: layout.align(),
                }
            }
        };
        Self {
            kind,
            source: Some(source),
        }
    }

    /// Returns the stable allocation-failure classification.
    #[must_use]
    pub const fn kind(&self) -> NativeMapAllocationKind {
        self.kind
    }
}

impl std::fmt::Display for NativeMapAllocationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            NativeMapAllocationKind::CapacityOverflow => {
                formatter.write_str("computed native map capacity exceeded the collection maximum")
            }
            NativeMapAllocationKind::AllocatorRefused {
                requested_bytes,
                alignment,
            } => write!(
                formatter,
                "allocator refused a {requested_bytes}-byte native map layout with {alignment}-byte alignment"
            ),
        }
    }
}

impl std::error::Error for NativeMapAllocationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_map_error_retains_dependency_source_behind_stable_kind() {
        let error =
            NativeMapAllocationError::from_hashbrown(hashbrown::TryReserveError::CapacityOverflow);

        assert_eq!(error.kind(), NativeMapAllocationKind::CapacityOverflow);
        assert!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<hashbrown::TryReserveError>())
                .is_some()
        );
    }

    #[test]
    fn native_map_allocator_refusal_preserves_layout_classification() {
        let layout = Layout::from_size_align(4096, 64).unwrap();
        let error =
            NativeMapAllocationError::from_hashbrown(hashbrown::TryReserveError::AllocError {
                layout,
            });

        assert_eq!(
            error.kind(),
            NativeMapAllocationKind::AllocatorRefused {
                requested_bytes: 4096,
                alignment: 64,
            }
        );
        assert!(std::error::Error::source(&error).is_some());
    }
}
