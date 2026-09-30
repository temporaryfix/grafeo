//! Explicit low-level access for engine durability and recovery fixtures.
//!
//! Application code must use `GrafeoDB` or `Session` APIs instead.

use std::sync::Arc;

use grafeo_core::graph::lpg::LpgStore;

use super::GrafeoDB;

/// Returns the concrete root LPG store for engine-internal test fixtures.
///
/// This preserves the retained-`Arc` and pointer-identity checks used by
/// durability, recovery, and hostile raw-mutation tests without exposing the
/// raw store as an inherent `GrafeoDB` application API.
#[doc(hidden)]
#[must_use]
pub fn root_lpg_store(db: &GrafeoDB) -> &Arc<LpgStore> {
    db.store_arc()
}

#[cfg(all(
    test,
    feature = "spill",
    feature = "gql",
    feature = "async-storage",
    any(target_os = "linux", target_os = "macos")
))]
pub(crate) fn install_spill_io(db: &GrafeoDB, io: Arc<dyn grafeo_core::execution::spill::SpillIo>) {
    db.spill_root.install_test_io(io);
}
