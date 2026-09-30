//! The core types you'll work with in Grafeo.
//!
//! Most of these are re-exported from the main `grafeo` crate, so you rarely
//! need to import from here directly.
//!
//! - **IDs**: [`NodeId`], [`EdgeId`] - handles to graph elements
//! - **Values**: [`Value`] - the dynamic type for properties
//! - **Keys**: [`PropertyKey`] - interned property names
//! - **Time**: [`Timestamp`] - for temporal properties

mod content_id;
mod date;
mod durable_cursor;
mod duration;
mod entity_index;
mod graph_path;
pub mod graph_path_bytes;
mod history_id;
pub mod hlc;
mod id;
mod logical_type;
mod property_map;
mod time;
mod timestamp;
mod valid_time;
mod validity;
mod value;
mod version_chain;
mod world_cut;
mod zoned_datetime;

pub use content_id::{ContentId, EntityRef, EvidenceRef, Interner, LocalId};
pub use date::Date;
pub use durable_cursor::{DurableCursor, FeedId};
pub use duration::Duration;
pub use entity_index::EntityIndex;
pub use graph_path::{GraphPath, GraphPathError, MAX_GRAPH_PATH_COMPONENTS};
pub use history_id::{
    GraphIncarnationId, HistoryCompleteness, InvalidStoreId, ParseStableIdError, StatementHandle,
    StoreId, StoreIdGenerationError,
};
pub use hlc::{HlcClock, HlcTimestamp};
pub use id::{EdgeId, EdgeTypeId, EpochId, IndexId, LabelId, NodeId, PropertyKeyId, TransactionId};
pub use logical_type::LogicalType;
pub use property_map::PropertyMap;
pub use time::Time;
pub use timestamp::Timestamp;
pub use valid_time::{InvalidValidTimeInterval, TaiNanoseconds, ValidTimeInterval};
pub use validity::ValidityTs;
pub use value::{
    HashableValue, INTERNAL_RDF_TAGGED_TERM_MARKER, OrderableValue, OrderedFloat64, PropertyKey,
    Value, canonical_f64_bits,
};
pub use version_chain::{EpochInterval, EpochZoneMap, VersionChain};
pub use world_cut::{
    AuthoritativeFormat, Digest256, GraphModelTag, MAX_RECOVERY_IMAGE_COMPONENTS,
    MAX_WORLD_GRAPH_NAME_BYTES, ModelFormatVersion, ProjectionCut, ProjectionReconciliationState,
    ProjectionSourceGraph, RecoveryImageComponent, RecoveryImageCoordinatesV1, RecoveryImageDigest,
    SchemaCut, SnapshotArtifact, StateDigest, StateDigestKind, WorldCut, WorldCutDescriptor,
    WorldCutError, WorldIdentityMetadataV1, WorldMetadataSectionV1, WorldMetadataSectionV2,
};
pub use zoned_datetime::ZonedDatetime;

// Re-export ArcStr so downstream crates don't need a direct arcstr dependency.
pub use arcstr::ArcStr;
