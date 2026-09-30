//! Vectorized query execution engine.
//!
//! Grafeo uses vectorized processing - instead of one row at a time, we process
//! batches of ~1024 rows. This unlocks SIMD and keeps the CPU busy.
//!
//! | Module | Purpose |
//! | ------ | ------- |
//! | [`chunk`] | Batched rows (DataChunk = multiple columns) |
//! | [`vector`] | Single column of values |
//! | [`factorized_vector`] | Factorized vectors for avoiding Cartesian products |
//! | [`factorized_chunk`] | Multi-level factorized chunks |
//! | [`selection`] | Bitmap for filtering without copying |
//! | [`operators`] | Physical operators (scan, filter, join, etc.) |
//! | [`pipeline`] | Push-based execution (data flows through operators) |
//! | [`parallel`] | Morsel-driven parallelism |
//! | [`spill`] | Disk spilling when memory is tight |
//! | [`adaptive`] | Adaptive execution with runtime cardinality feedback |
//! | [`collector`] | Generic collector pattern for parallel aggregation |
//!
//! The execution model is push-based: sources push data through a pipeline of
//! operators until it reaches a sink.

mod accounted_chunk;
pub mod adaptive;
pub mod cancellation;
pub mod chunk;
pub mod chunk_state;
pub mod collector;
pub mod factorized_chunk;
pub mod factorized_iter;
pub mod factorized_vector;
pub mod memory;
mod native_allocation;
pub mod operators;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod pipeline;
pub mod pipeline_convert;
pub mod profile;
pub mod scheduler;
pub mod selection;
pub mod sink;
pub mod source;
#[cfg(feature = "spill")]
pub mod spill;
#[doc(hidden)]
pub mod value_codec;
pub mod vector;

pub use accounted_chunk::AccountedDataChunk;
pub use adaptive::{
    AdaptiveCheckpoint, AdaptiveContext, AdaptiveEvent, AdaptiveExecutionConfig,
    AdaptiveExecutionResult, AdaptivePipelineBuilder, AdaptivePipelineConfig,
    AdaptivePipelineExecutor, AdaptiveSummary, CardinalityCheckpoint, CardinalityFeedback,
    CardinalityTrackingOperator, CardinalityTrackingSink, CardinalityTrackingWrapper,
    ReoptimizationDecision, SharedAdaptiveContext, evaluate_reoptimization, execute_adaptive,
};
pub use cancellation::{
    QueryCancellationError, QueryCancellationHandle, QueryCancellationToken, QueryDeadlineError,
    QueryExecutionCheckpoint, QueryExecutionControl, QueryLifecycleError,
};
pub use chunk::{ChunkZoneHints, DataChunk};
pub use collector::{
    Collector, CollectorStats, CountCollector, LimitCollector, MaterializeCollector,
    PartitionCollector, StatsCollector,
};
pub use memory::{
    ExecutionMemoryContext, ExecutionMemoryContextBuilder, QueryExecutionId, QueryResourceContext,
    QueryResourceContextError, QueryResourceStats,
};
pub use native_allocation::{NativeMapAllocationError, NativeMapAllocationKind};
#[cfg(feature = "parallel")]
pub use parallel::{
    CloneableOperatorFactory, MorselScheduler, ParallelPipeline, ParallelPipelineConfig,
    ParallelSource, RangeSource,
};
pub use pipeline::{
    ChunkCollector, ChunkSizeHint, ChunkTransport, Pipeline, PushOperator, Sink, Source,
    VariantCollector,
};
pub use profile::{ProfileStats, ProfiledOperator, SharedProfileStats};
pub use scheduler::Scheduler;
pub use selection::SelectionVector;
pub use sink::{CollectorSink, CountingSink, LimitingSink, MaterializingSink, NullSink};
pub use source::{ChunkSource, EmptySource, GeneratorSource, OperatorSource, VectorSource};
#[cfg(feature = "spill")]
pub use spill::{
    CleartextSpillRecordProvider, NoopSpillIo, OpenSpillRecord, SpillDiskQuota, SpillDiskStats,
    SpillFile, SpillFileIdentity, SpillFileReader, SpillFileRole, SpillFrameLimits, SpillIo,
    SpillIoOperation, SpillManager, SpillQueryIdentity, SpillQuotaExceeded, SpillRecordKind,
    SpillRecordMeta, SpillRecordProvider,
};
pub use vector::{ResidentCapacityError, ValueVector};

// Factorized execution types
pub use chunk_state::{ChunkState, FactorizationState, FactorizedSelection, LevelSelection};
pub use factorized_chunk::{ChunkVariant, FactorizationLevel, FactorizedChunk};
pub use factorized_iter::{PrecomputedIter, RowIndices, RowView, StreamingIter};
pub use factorized_vector::{FactorizedState, FactorizedVector, UnflatMetadata};
pub use operators::{FactorizedData, FlatDataWrapper};

#[cfg(all(test, not(feature = "spill")))]
mod value_codec_profile_tests {
    use super::value_codec::{deserialize_row, serialize_row};
    use grafeo_common::types::Value;
    use std::io::Cursor;

    #[test]
    fn value_codec_is_available_without_disk_spill() {
        let row = [Value::RdfLiteral {
            lexical: "18446744073709551616".into(),
            language: None,
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
        }];
        let mut encoded = Vec::new();

        serialize_row(&row, &mut encoded).unwrap();

        assert_eq!(deserialize_row(&mut Cursor::new(encoded), 1).unwrap(), row);
    }
}
