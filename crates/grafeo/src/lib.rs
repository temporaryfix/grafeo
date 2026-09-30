//! # Grafeo
//!
//! A high-performance, embeddable graph database with a Rust core and no required
//! C dependencies. Optional allocators (jemalloc/mimalloc) and TLS use C libraries
//! for performance.
//!
//! If you're new here, start with [`GrafeoDB`] - that's your entry point for
//! creating databases and running queries. Grafeo uses GQL (the ISO standard)
//! by default, but you can enable other query languages through feature flags.
//!
//! ## Query Languages
//!
//! | Feature | Language | Notes |
//! | ------- | -------- | ----- |
//! | `gql` | GQL | ISO standard, enabled by default |
//! | `cypher` | Cypher | Neo4j-style queries |
//! | `sparql` | SPARQL | For RDF triple stores |
//! | `gremlin` | Gremlin | Apache TinkerPop traversals |
//! | `graphql` | GraphQL | Schema-based queries |
//! | `sql-pgq` | SQL/PGQ | SQL:2023 GRAPH_TABLE |
//!
//! Compose the persona features (`lpg`, `rdf`, `analytics`, and `ai`) for the
//! capabilities an application needs.
//!
//! ## Quick Start
//!
//! ```rust
//! use grafeo::GrafeoDB;
//!
//! // Create an in-memory database
//! let db = GrafeoDB::new_in_memory();
//! let mut session = db.session();
//!
//! // Add a person
//! session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
//!
//! // Find them
//! let result = session.execute("MATCH (p:Person) RETURN p.name")?;
//! # Ok::<(), grafeo::Error>(())
//! ```
//!
//! ## Performance Features
//!
//! Enable platform-optimized memory allocators for 10-20% faster allocations:
//!
//! - `jemalloc` - Linux/macOS (x86_64, aarch64)
//! - `mimalloc-allocator` - Windows
//!
#![forbid(unsafe_code)]

// Named profiles are public capability contracts. Keep these compile-time
// guards beside the facade so a manifest-only edit cannot silently turn a
// comprehensive profile into a smaller build. Users who need minimal binaries
// can compose the atomic features directly.
#[cfg(all(
    feature = "rdf",
    not(all(
        feature = "triple-store",
        feature = "sparql",
        feature = "graphql",
        feature = "ring-index",
        feature = "storage",
        feature = "regex",
        feature = "shacl"
    ))
))]
compile_error!(
    "the `rdf` profile must retain its complete query/index/validation/storage contract"
);

#[cfg(all(
    feature = "edge",
    not(all(feature = "gql", feature = "compact-store", feature = "regex-lite"))
))]
compile_error!("the `edge` profile must retain LPG/GQL/compact-store/regex-lite");

// Platform-optimized memory allocators (enabled via features)
// jemalloc: Linux/macOS x86_64/aarch64 - better multi-threaded performance
// mimalloc: Windows - optimized for Windows, better than MSVC allocator
#[cfg(all(
    feature = "jemalloc",
    not(target_os = "windows"),
    not(target_os = "openbsd"),
    not(target_env = "musl"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc-allocator", target_os = "windows"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Re-export the main database API
pub use grafeo_engine::{
    AccessMode, Catalog, CatalogError, Config, ConfigError, Direction, DurabilityMode, GrafeoDB,
    Grant, GraphModel, GraphStore, GraphStoreMut, Identity, IndexDefinition, IndexType,
    IsolationLevel, PhysicalIndexFamily, PhysicalIndexKey, Role, Session, StatementKind,
    TransactionManagerView, VERSION,
};
#[cfg(feature = "statement-table")]
pub use grafeo_engine::{StatementIngest, StatementRow};

// Re-export submodules for qualified access (e.g. grafeo::auth::Identity)
pub use grafeo_engine::admin;
pub use grafeo_engine::auth;
#[cfg(feature = "cdc")]
pub use grafeo_engine::cdc;
pub use grafeo_engine::database;
pub use grafeo_engine::memory_usage;
pub use grafeo_engine::session;

// Re-export query results
pub use grafeo_engine::database::QueryResult;

// Re-export MemoryUsage, ProjectionSpec at top level for convenience
pub use grafeo_engine::MemoryUsage;
pub use grafeo_engine::ProjectionSpec;

// Re-export core types - you'll need these for working with IDs and values
pub use grafeo_common::types::{
    ContentId, EdgeId, EpochId, GraphPath, IndexId, NodeId, PropertyKey, Value,
};
#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
pub use grafeo_engine::CompactStoreTieredView;
#[cfg(feature = "grafeo-file")]
pub use grafeo_engine::DatabaseFileView;
#[cfg(all(feature = "compact-store", feature = "lpg"))]
pub use grafeo_engine::LayeredStoreView;
pub use grafeo_engine::{
    AuthoritativeFormat, Digest256, GraphIncarnationId, GraphModelTag, HistoryCompleteness,
    InvalidValidTimeInterval, MAX_RECOVERY_IMAGE_COMPONENTS, ModelFormatVersion, ProjectionCut,
    ProjectionReconciliationState, ProjectionSourceGraph, RecoveryImageComponent,
    RecoveryImageCoordinatesV1, RecoveryImageDigest, SchemaCut, SnapshotArtifact, StateDigest,
    StateDigestKind, StatementHandle, StoreId, TaiNanoseconds, ValidTimeInterval, WorldCut,
    WorldCutDescriptor, WorldCutError, WorldIdentityMetadataV1, WorldMetadataSectionV1,
    WorldMetadataSectionV2,
};
#[cfg(any(
    feature = "lpg",
    feature = "lpg-model",
    feature = "edge",
    feature = "temporal-host",
    feature = "native"
))]
pub use grafeo_engine::{CreateIndexRequest, IndexCreateKind};
#[cfg(feature = "compact-store")]
pub use grafeo_engine::{GraphScrub, NodeTableScrub, RelTableScrub};
#[cfg(feature = "triple-store")]
pub use grafeo_engine::{
    Quad, RdfCdcPage, RdfDatasetHistory, RdfGraphIdentity, RdfGraphLife, RdfHistoricalQuad,
    RdfHistoryCursor, RdfHistoryCursorError, RdfHistoryCut, RdfHistoryDiff, RdfHistoryError,
    RdfHistoryTransition, RdfHistoryTransitionKind, RdfQuadVersion, Term, Triple,
};

// Re-export error types so users don't need to depend on grafeo-common directly
pub use grafeo_common::utils::error::{Error, Result};
