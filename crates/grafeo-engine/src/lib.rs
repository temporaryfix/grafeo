//! # grafeo-engine
//!
//! The engine behind Grafeo. You'll find everything here for creating databases,
//! running queries, and managing transactions.
//!
//! Most users should start with the main `grafeo` crate, which re-exports the
//! key types. If you're here directly, [`GrafeoDB`] is your entry point.
//!
//! ## Modules
//!
//! - [`database`] - Create and manage databases with [`GrafeoDB`]
//! - [`session`] - Lightweight handles for concurrent access
//! - [`config`] - Tune memory, threads, and durability settings
//! - [`transaction`] - MVCC transaction management (snapshot isolation)
//! - [`query`] - The full query pipeline: parsing, planning, optimization, execution
//! - [`catalog`] - Schema metadata: labels, property keys, indexes
//! - [`admin`] - Admin API types for inspection, backup, and maintenance

#![deny(unsafe_code)]
// Parser-free `native` (no GQL/SPARQL/Cypher) still compiles translators,
// language helpers, and their unit tests as dead code. Those lints are
// meaningful when a query language is on; they are noise on this profile.
#![cfg_attr(
    not(any(feature = "gql", feature = "cypher", feature = "sparql")),
    allow(
        dead_code,
        unreachable_code,
        unreachable_patterns,
        unused_assignments,
        unused_imports,
        unused_variables,
        clippy::bool_to_int_with_if,
        clippy::collapsible_if,
        clippy::if_same_then_else,
        clippy::items_after_test_module,
        clippy::manual_assert_eq,
        clippy::missing_errors_doc,
        clippy::missing_panics_doc,
        clippy::needless_borrow,
        clippy::needless_borrows_for_generic_args,
        clippy::needless_lifetimes,
        clippy::option_option,
        clippy::question_mark,
        clippy::too_many_arguments,
        clippy::useless_borrows_in_formatting,
        clippy::useless_format
    )
)]

/// The version of the grafeo-engine crate (from Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
#[path = "../../grafeo-core/tests/support/allocation.rs"]
pub(crate) mod allocation_test;

pub mod admin;
pub mod auth;
pub mod catalog;
#[cfg(feature = "cdc")]
pub mod cdc;
pub mod config;
pub mod database;
#[cfg(feature = "embed")]
pub mod embedding;
pub mod execution;
pub mod export;
pub mod memory_usage;
#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(any(feature = "lpg", feature = "algos"))]
pub mod procedures;
pub mod query;
pub mod session;
#[cfg(feature = "spill")]
mod spill_crypto;
pub mod transaction;
#[cfg(feature = "shacl")]
pub mod validation;

pub use admin::{
    AdminService, CompactionStats, DatabaseInfo, DatabaseMode, DatabaseStats, DumpFormat,
    DumpMetadata, IndexInfo, LpgSchemaInfo, RdfSchemaInfo, SchemaInfo, ValidationError,
    ValidationResult, ValidationWarning, WalStatus,
};
pub use auth::{Grant, Identity, Role, StatementKind};
pub use catalog::{Catalog, CatalogError, IndexDefinition, IndexType};
pub use config::{AccessMode, Config, ConfigError, DurabilityMode, GraphModel};
#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
pub use database::CompactStoreTieredView;
#[cfg(feature = "grafeo-file")]
pub use database::DatabaseFileView;
pub use database::GrafeoDB;
#[cfg(all(feature = "compact-store", feature = "lpg"))]
pub use database::LayeredStoreView;
#[cfg(feature = "wal")]
pub use database::WalControl;
#[cfg(feature = "lpg")]
pub use database::{CreateIndexRequest, IndexCreateKind};
#[cfg(feature = "lpg")]
pub use database::{
    IndexMergePolicy, OpenMultiOptions, SchemaMergePolicy, SnapshotInfo, snapshot_info,
};
pub use grafeo_common::types::{
    AuthoritativeFormat, Digest256, GraphIncarnationId, GraphModelTag, GraphPath,
    HistoryCompleteness, IndexId, InvalidValidTimeInterval, MAX_RECOVERY_IMAGE_COMPONENTS,
    ModelFormatVersion, ProjectionCut, ProjectionReconciliationState, ProjectionSourceGraph,
    RecoveryImageComponent, RecoveryImageCoordinatesV1, RecoveryImageDigest, SchemaCut,
    SnapshotArtifact, StateDigest, StateDigestKind, StatementHandle, StoreId, TaiNanoseconds,
    ValidTimeInterval, WorldCut, WorldCutDescriptor, WorldCutError, WorldIdentityMetadataV1,
    WorldMetadataSectionV1, WorldMetadataSectionV2,
};
#[cfg(feature = "statement-table")]
pub use grafeo_core::graph::compact::statement_table::{
    StatementIngest, StatementRow, StatementTable, StatementTableInsertError,
};
#[cfg(feature = "compact-store")]
pub use grafeo_core::graph::compact::{GraphScrub, NodeTableScrub, RelTableScrub};
pub use grafeo_core::graph::lpg::{PhysicalIndexFamily, PhysicalIndexKey};
#[cfg(feature = "triple-store")]
pub use grafeo_core::graph::rdf::{
    Quad, RdfCdcPage, RdfDatasetHistory, RdfGraphIdentity, RdfGraphLife, RdfHistoricalQuad,
    RdfHistoryCursor, RdfHistoryCursorError, RdfHistoryCut, RdfHistoryDiff, RdfHistoryError,
    RdfHistoryTransition, RdfHistoryTransitionKind, RdfQuadVersion, Term, Triple,
};
pub use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut, ProjectionSpec};
pub use memory_usage::MemoryUsage;
#[cfg(feature = "metrics")]
pub use metrics::{MetricsRegistry, MetricsSnapshot};
#[cfg(all(feature = "gql", feature = "lpg"))]
pub use query::executor::stream::{
    OwnedResultStream, OwnedRowIterator, ResultStream, RowIterator, StreamChunk,
};
pub use query::{ExecutionOptions, ResultLimits};
#[cfg(feature = "testing-statement-injection")]
pub use session::QueryCancellationTestPhase;
pub use session::{MixedSnapshot, Session};
#[cfg(feature = "lpg")]
pub use transaction::{CommitInfo, PreparedCommit};
pub use transaction::{ConflictGranularity, IsolationLevel, TransactionManagerView};
