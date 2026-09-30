//! Labeled Property Graph (LPG) storage.
//!
//! This is Grafeo's primary graph model - the same model used by Neo4j,
//! TigerGraph, and most modern graph databases. If you're used to working
//! with nodes, relationships, and properties, you're in the right place.
//!
//! ## What you get
//!
//! - **Nodes** with labels (like "Person", "Company") and properties (like "name", "age")
//! - **Edges** that connect nodes, with types (like "KNOWS", "WORKS_AT") and their own properties
//! - **Indexes** that make lookups fast
//!
//! Start with [`LpgStore`] - that's where everything lives.

mod edge;
pub mod exact_history;
mod index_key;
mod node;
pub mod overlay;
mod property;
#[cfg(feature = "lpg")]
pub mod section;
#[cfg(feature = "lpg")]
mod store;
#[cfg(feature = "lpg")]
mod value_codec;

// Types are always available (used by GraphStore trait and RDF adapter)
pub use edge::{Edge, EdgeFlags, EdgeRecord};
pub use index_key::{
    PhysicalIndexFamily, PhysicalIndexKey, canonicalize_index_key, canonicalize_scoped_index_key,
    decode_index_key, decode_scoped_index_key, encode_index_key, encode_scoped_index_key,
    index_keys_match, scoped_index_keys_match,
};
pub use node::{Node, NodeFlags, NodeRecord};
pub use property::{CompareOp, PropertyStorage};

// Store and section require the lpg feature
#[cfg(feature = "lpg")]
pub use section::{LpgSectionCapture, LpgStoreSection};
#[cfg(feature = "lpg")]
pub(crate) use store::DataCommitScope;
#[cfg(all(feature = "lpg", any(test, feature = "compact-store")))]
pub(crate) use store::PinnedLpgTransition;
#[cfg(all(feature = "lpg", feature = "vector-index", feature = "compact-store"))]
pub(crate) use store::VisibleVectorReadContext;
#[cfg(feature = "lpg")]
pub use store::{
    DataRebindError, IndexRegistrationObservation, IndexRegistryContents, IndexRegistryEdit,
    IndexRegistryKey, IndexRegistryMaintenance, IndexRegistryWorkspace,
    InstalledIndexRegistryFence, InstalledLpgCommit, LpgCommitWorkspace, LpgReplacementWorkspace,
    LpgStore, LpgStoreConfig, PreparedIndexRegistryBatch, PreparedLpgCommit,
    PreparedNodeLabelImages, PropertyIndexImage, PropertyUndoEntry, ReleasedLpgCommit,
    StoreCommitInput, StoreIndexEdits, TransportEdgeMutationGrant, TransportEdgeReceipt,
    TransportEdgeState, TxDelta, prepare_index_registry_batch, with_prepared_lpg_commit,
    with_prepared_lpg_replacement,
};
#[cfg(all(feature = "lpg", feature = "compact-store"))]
pub(crate) use store::{
    PinnedNamedGraphTopology, PreparedPurgeOutcome, PreparedRepresentationTransfer,
    PublishedRepresentationTransfer,
};
#[cfg(all(feature = "lpg", feature = "vector-index"))]
pub use store::{VectorCommitChanges, VectorCommitInput};
#[cfg(feature = "lpg")]
#[doc(hidden)]
pub use value_codec::{decode_value_exact, encode_value};
