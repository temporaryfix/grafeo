//! Push-based execution pipeline.
//!
//! This module provides push-based execution primitives where data flows
//! forward through operators via `push()` calls, enabling better parallelism
//! and cache utilization compared to pull-based execution.

use super::accounted_chunk::AccountedDataChunk;
use super::cancellation::QueryExecutionCheckpoint;
use super::chunk::DataChunk;
use super::factorized_chunk::ChunkVariant;
use super::operators::OperatorError;

/// Hint for preferred chunk size.
///
/// Operators can provide hints to optimize chunk sizing for their workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkSizeHint {
    /// Use default chunk size (2048 tuples).
    Default,
    /// Use small chunks (256-512 tuples) for LIMIT or high selectivity.
    Small,
    /// Use large chunks (4096 tuples) for full scans.
    Large,
    /// Use exact chunk size.
    Exact(usize),
    /// Use at most this many tuples (for LIMIT).
    AtMost(usize),
}

impl Default for ChunkSizeHint {
    fn default() -> Self {
        Self::Default
    }
}

/// Default chunk size in tuples.
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

/// Small chunk size for high selectivity or LIMIT.
pub const SMALL_CHUNK_SIZE: usize = 512;

/// Large chunk size for full scans.
pub const LARGE_CHUNK_SIZE: usize = 4096;

/// Move-only chunk transports that may reach a pipeline stage.
///
/// Admission is monotone: once a stage may emit an accounted chunk, every
/// later stage must preserve that possibility. This describes capability,
/// not a promise that every chunk in the pipeline is accounted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkTransport {
    /// Only compatibility [`DataChunk`] values can reach the stage.
    #[default]
    PlainOnly,
    /// Either a compatibility chunk or a move-only accounted envelope can
    /// reach the stage.
    MayBeAccounted,
}

impl ChunkTransport {
    const fn union(self, other: Self) -> Self {
        if matches!(self, Self::MayBeAccounted) || matches!(other, Self::MayBeAccounted) {
            Self::MayBeAccounted
        } else {
            Self::PlainOnly
        }
    }
}

pub(in crate::execution) mod qualified_accounted_transport {
    use super::{AccountedDataChunk, AccountedSinkPermit, OperatorError, PushOperator, Sink};

    /// Crate-sealed implementation behind a sink permit.
    pub(in crate::execution) trait QualifiedSink: Sink {
        fn consume_accounted_qualified(
            &mut self,
            chunk: AccountedDataChunk,
        ) -> Result<bool, OperatorError>;
    }

    /// Crate-sealed implementation behind an operator permit.
    pub(in crate::execution) trait QualifiedPushOperator:
        PushOperator
    {
        fn push_accounted_qualified(
            &mut self,
            chunk: AccountedDataChunk,
            sink: &mut AccountedSinkPermit<'_>,
        ) -> Result<bool, OperatorError>;
    }
}

/// Borrowed authority to invoke one exact, crate-audited accounted sink.
///
/// Public trait implementations can relay a permit borrowed from an audited
/// inner sink, but safe downstream code cannot construct or invoke one. The
/// invocation therefore reaches the object bound into this permit rather than
/// an untrusted public [`Sink::consume_accounted`] override.
///
/// ```compile_fail
/// use grafeo_core::execution::{AccountedDataChunk, pipeline::AccountedSinkPermit};
///
/// fn invoke(
///     permit: &mut AccountedSinkPermit<'_>,
///     chunk: AccountedDataChunk,
/// ) {
///     let _ = permit.consume(chunk);
/// }
/// ```
#[doc(hidden)]
pub struct AccountedSinkPermit<'a> {
    inner: &'a mut dyn qualified_accounted_transport::QualifiedSink,
}

impl<'a> AccountedSinkPermit<'a> {
    pub(in crate::execution) fn new(
        inner: &'a mut dyn qualified_accounted_transport::QualifiedSink,
    ) -> Self {
        Self { inner }
    }

    pub(in crate::execution) fn consume(
        &mut self,
        chunk: AccountedDataChunk,
    ) -> Result<bool, OperatorError> {
        self.inner.consume_accounted_qualified(chunk)
    }
}

/// Borrowed authority to invoke one exact, crate-audited accounted operator.
///
/// ```compile_fail
/// use grafeo_core::execution::{
///     AccountedDataChunk,
///     pipeline::{AccountedPushPermit, AccountedSinkPermit},
/// };
///
/// fn invoke(
///     operator: &mut AccountedPushPermit<'_>,
///     chunk: AccountedDataChunk,
///     sink: &mut AccountedSinkPermit<'_>,
/// ) {
///     let _ = operator.push(chunk, sink);
/// }
/// ```
#[doc(hidden)]
pub struct AccountedPushPermit<'a> {
    inner: &'a mut dyn qualified_accounted_transport::QualifiedPushOperator,
}

impl<'a> AccountedPushPermit<'a> {
    pub(in crate::execution) fn new(
        inner: &'a mut dyn qualified_accounted_transport::QualifiedPushOperator,
    ) -> Self {
        Self { inner }
    }

    pub(in crate::execution) fn push(
        &mut self,
        chunk: AccountedDataChunk,
        sink: &mut AccountedSinkPermit<'_>,
    ) -> Result<bool, OperatorError> {
        self.inner.push_accounted_qualified(chunk, sink)
    }
}

/// Source of data chunks for a pipeline.
///
/// Sources produce chunks of data that flow through the pipeline.
/// With an explicit cooperative checkpoint, any successful fallible call may
/// be followed immediately by cancellation. Externally visible effects must
/// remain rollbackable until the execution owner fences success.
pub trait Source: Send + Sync {
    /// Produce the next chunk of data.
    ///
    /// Returns `Ok(Some(chunk))` if data is available, `Ok(None)` if exhausted.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the source fails to produce data.
    /// Under an explicit cooperative checkpoint, a successful call may be
    /// followed immediately by cancellation; externally visible effects must
    /// remain rollbackable until the execution owner fences success.
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError>;

    /// Reset the source to its initial state.
    fn reset(&mut self);

    /// Name of this source for debugging.
    fn name(&self) -> &'static str;
}

/// Sink that receives output from operators.
///
/// Sinks consume data chunks produced by the pipeline.
/// With an explicit cooperative checkpoint, any successful fallible call,
/// including finalization, may be followed immediately by cancellation. Such
/// effects must remain rollbackable, and cleanup must remain available through
/// owner rollback or `Drop`, until the execution owner fences success.
pub trait Sink: Send + Sync {
    /// Consume a chunk of data.
    ///
    /// Returns `Ok(true)` to continue, `Ok(false)` to signal terminal early
    /// termination. After `Ok(false)`, the pipeline will not consume another
    /// chunk, but it will still call the terminal sink's [`Self::finalize`]
    /// once for lifecycle cleanup.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the sink fails to process the chunk.
    /// Under an explicit cooperative checkpoint, a successful call may be
    /// followed immediately by cancellation; externally visible effects must
    /// remain rollbackable until the execution owner fences success.
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError>;

    /// Consume a move-only chunk together with its resident-memory authority.
    ///
    /// The default rejects without cloning, unwrapping, or calling
    /// [`Self::consume`]. Implementations may opt in only when they preserve
    /// the envelope for the complete lifetime of any retained data.
    ///
    /// # Errors
    ///
    /// Returns [`OperatorError::UnsupportedAccountedTransport`] unless the
    /// sink explicitly supports accounted ownership.
    fn consume_accounted(&mut self, _chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
        Err(OperatorError::UnsupportedAccountedTransport {
            consumer: self.name(),
        })
    }

    /// Returns invocation authority only when this exact sink, or an audited
    /// inner sink it deliberately exposes, implements the crate-sealed lane.
    ///
    /// Public admission methods remain advisory and never mint this authority.
    #[doc(hidden)]
    fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
        None
    }

    /// Consume a flat or factorized chunk. Default flattens.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the sink fails to process the chunk.
    fn consume_variant(&mut self, chunk: ChunkVariant) -> Result<bool, OperatorError> {
        self.consume(chunk.ensure_flat())
    }

    /// Validates the set of chunk transports that may reach this sink.
    ///
    /// The default preserves all existing plain behavior and rejects a
    /// possible accounted envelope before source execution begins.
    ///
    /// # Errors
    ///
    /// Returns [`OperatorError::UnsupportedAccountedTransport`] when
    /// `input` may contain accounted chunks.
    fn admit_chunk_transport(&self, input: ChunkTransport) -> Result<(), OperatorError> {
        if input == ChunkTransport::MayBeAccounted {
            return Err(OperatorError::UnsupportedAccountedTransport {
                consumer: self.name(),
            });
        }
        Ok(())
    }

    /// Called once when pipeline execution ends, including after early
    /// termination. This is a lifecycle hook and must not emit more rows.
    ///
    /// # Errors
    ///
    /// Returns `Err` if finalization fails.
    fn finalize(&mut self) -> Result<(), OperatorError>;

    /// Name of this sink for debugging.
    fn name(&self) -> &'static str;

    /// Converts this boxed sink into `Box<dyn Any>` for type-based dispatch.
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

/// Push-based operator trait.
///
/// Unlike pull-based operators that return data on `next()` calls,
/// push-based operators receive data via `push()` and forward results
/// to a downstream sink.
/// With an explicit cooperative checkpoint, any successful fallible call,
/// including finalization, may be followed immediately by cancellation. Such
/// effects must remain rollbackable, and cleanup must remain available through
/// owner rollback or `Drop`, until the execution owner fences success.
pub trait PushOperator: Send + Sync {
    /// Process an incoming chunk and push results to the sink.
    ///
    /// Returns `Ok(true)` to continue processing, `Ok(false)` for terminal
    /// early termination. A stopped pipeline does not call operator finalizers,
    /// because those finalizers may emit additional rows; owned resources must
    /// therefore also remain reclaimable through `Drop`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operator or sink fails during processing.
    /// Under an explicit cooperative checkpoint, a successful call may be
    /// followed immediately by cancellation; externally visible effects must
    /// remain rollbackable until the execution owner fences success.
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError>;

    /// Process a move-only chunk without detaching it from its memory grant.
    ///
    /// The default rejects without cloning, unwrapping, or invoking
    /// [`Self::push`]. A preserving implementation forwards the same envelope
    /// or retains it with its grant.
    ///
    /// # Errors
    ///
    /// Returns [`OperatorError::UnsupportedAccountedTransport`] unless the
    /// operator explicitly supports accounted ownership.
    fn push_accounted(
        &mut self,
        _chunk: AccountedDataChunk,
        _sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        Err(OperatorError::UnsupportedAccountedTransport {
            consumer: self.name(),
        })
    }

    /// Returns invocation authority only when this exact operator, or an
    /// audited inner operator it deliberately exposes, implements the
    /// crate-sealed lane.
    ///
    /// Public admission methods remain advisory and never mint this authority.
    #[doc(hidden)]
    fn __accounted_push_permit(&mut self) -> Option<AccountedPushPermit<'_>> {
        None
    }

    /// Process a flat or factorized chunk. Default flattens then [`Self::push`].
    ///
    /// Operators that can stay factorized override this and
    /// [`Self::accepts_factorized`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operator or sink fails during processing.
    fn push_variant(
        &mut self,
        chunk: ChunkVariant,
        sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        self.push(chunk.ensure_flat(), sink)
    }

    /// True when [`Self::push_variant`] can keep a factorized chunk unflat.
    fn accepts_factorized(&self) -> bool {
        false
    }

    /// Validates and declares the chunk transports leaving this operator.
    ///
    /// Existing operators accept plain input and remain plain. The default
    /// rejects possible accounted input; preserving forwarders and qualified
    /// producers must opt in explicitly. Pipeline folding remains monotone
    /// even if an implementation returns a narrower declaration.
    ///
    /// # Errors
    ///
    /// Returns [`OperatorError::UnsupportedAccountedTransport`] when
    /// `input` may contain accounted chunks.
    fn admit_chunk_transport(
        &self,
        input: ChunkTransport,
    ) -> Result<ChunkTransport, OperatorError> {
        if input == ChunkTransport::MayBeAccounted {
            return Err(OperatorError::UnsupportedAccountedTransport {
                consumer: self.name(),
            });
        }
        Ok(ChunkTransport::PlainOnly)
    }

    /// Called when all input has been processed.
    ///
    /// Pipeline breakers (Sort, Aggregate, etc.) emit their results here.
    ///
    /// # Errors
    ///
    /// Returns `Err` if finalization or downstream sink consumption fails.
    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError>;

    /// Hint for preferred chunk size.
    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    /// Name of this operator for debugging.
    fn name(&self) -> &'static str;
}

/// Execution pipeline connecting source, operators, and sink.
pub struct Pipeline {
    source: Box<dyn Source>,
    operators: Vec<Box<dyn PushOperator>>,
    sink: Box<dyn Sink>,
    /// Explicit orchestration-only cancellation/deadline checkpoint.
    checkpoint: Option<QueryExecutionCheckpoint>,
    /// Origin of early termination, retained until downstream finalization.
    progress: ChainProgress,
}

/// Private continuation origin shared by serial and worker-local pipelines.
/// An exhausted producer cannot accept more input, but its downstream
/// breakers still own accepted rows that must be finalized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ChainProgress {
    #[default]
    Continue,
    ExhaustedAfter(usize),
    SinkStopped,
}

impl ChainProgress {
    pub(super) fn continued(self) -> bool {
        matches!(self, Self::Continue)
    }

    fn finalize_start(self, operator_count: usize) -> usize {
        match self {
            Self::Continue => 0,
            Self::ExhaustedAfter(index) => index.saturating_add(1),
            Self::SinkStopped => operator_count,
        }
    }
}

fn check_streaming_checkpoint(
    checkpoint: Option<&QueryExecutionCheckpoint>,
) -> Result<(), OperatorError> {
    checkpoint.map_or(Ok(()), |checkpoint| checkpoint.check().map_err(Into::into))
}

fn admit_chunk_transport_chain(
    operators: &[Box<dyn PushOperator>],
    sink: &dyn Sink,
    input: ChunkTransport,
) -> Result<(), OperatorError> {
    let transport = operators.iter().try_fold(input, |transport, operator| {
        let declared = operator.admit_chunk_transport(transport)?;
        Ok::<_, OperatorError>(transport.union(declared))
    })?;
    sink.admit_chunk_transport(transport)
}

fn accounted_sink_permit(sink: &mut dyn Sink) -> Result<AccountedSinkPermit<'_>, OperatorError> {
    let consumer = sink.name();
    sink.__accounted_sink_permit()
        .ok_or(OperatorError::UnsupportedAccountedTransport { consumer })
}

fn accounted_push_permit(
    operator: &mut dyn PushOperator,
) -> Result<AccountedPushPermit<'_>, OperatorError> {
    let consumer = operator.name();
    operator
        .__accounted_push_permit()
        .ok_or(OperatorError::UnsupportedAccountedTransport { consumer })
}

/// Repeats advisory validation while requiring an unforgeable permit at every
/// stage that may actually receive an accounted envelope.
fn admit_qualified_chunk_transport_chain(
    operators: &mut [Box<dyn PushOperator>],
    sink: &mut dyn Sink,
    input: ChunkTransport,
) -> Result<(), OperatorError> {
    let mut transport = input;
    for operator in operators {
        let declared = operator.admit_chunk_transport(transport)?;
        let output = transport.union(declared);
        if output == ChunkTransport::MayBeAccounted {
            let _permit = accounted_push_permit(operator.as_mut())?;
        }
        transport = output;
    }
    sink.admit_chunk_transport(transport)?;
    if transport == ChunkTransport::MayBeAccounted {
        let _permit = accounted_sink_permit(sink)?;
    }
    Ok(())
}

fn qualified_accounted_chain(operators: &mut [Box<dyn PushOperator>], sink: &mut dyn Sink) -> bool {
    for operator in operators {
        if operator.__accounted_push_permit().is_none() {
            return false;
        }
    }
    sink.__accounted_sink_permit().is_some()
}

/// Borrowed relay used only while one breaker is finalizing. Each emitted
/// chunk enters the remaining operator chain immediately, so no intermediate
/// collector can retain the breaker's complete result.
struct StreamingPipelineSink<'a> {
    operators: &'a mut [Box<dyn PushOperator>],
    sink: &'a mut dyn Sink,
    checkpoint: Option<&'a QueryExecutionCheckpoint>,
    stopped: bool,
    start_index: usize,
    progress: &'a mut ChainProgress,
}

/// Borrowed relay that owns the terminal continuation protocol while one
/// operator is pushing. It prevents an operator that ignores `false` from
/// invoking the real sink again, while preserving factorized chunks.
pub(super) struct TerminalContinuationSink<'a> {
    pub(super) sink: &'a mut dyn Sink,
    pub(super) stopped: bool,
}

impl Sink for TerminalContinuationSink<'_> {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        let continued = self.sink.consume(chunk)?;
        self.stopped = !continued;
        Ok(continued)
    }

    fn consume_variant(&mut self, chunk: ChunkVariant) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        let continued = self.sink.consume_variant(chunk)?;
        self.stopped = !continued;
        Ok(continued)
    }

    fn consume_accounted(&mut self, chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
        <Self as qualified_accounted_transport::QualifiedSink>::consume_accounted_qualified(
            self, chunk,
        )
    }

    fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
        let qualified = self.sink.__accounted_sink_permit().is_some();
        if qualified {
            Some(AccountedSinkPermit::new(self))
        } else {
            None
        }
    }

    fn admit_chunk_transport(&self, input: ChunkTransport) -> Result<(), OperatorError> {
        self.sink.admit_chunk_transport(input)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "TerminalContinuationSink"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        panic!("borrowed terminal continuation sink cannot escape a push")
    }
}

impl qualified_accounted_transport::QualifiedSink for TerminalContinuationSink<'_> {
    fn consume_accounted_qualified(
        &mut self,
        chunk: AccountedDataChunk,
    ) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        let continued = accounted_sink_permit(self.sink)?.consume(chunk)?;
        self.stopped = !continued;
        Ok(continued)
    }
}

impl Sink for StreamingPipelineSink<'_> {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        let continued = push_through_streaming_chain(
            self.operators,
            self.sink,
            self.checkpoint,
            chunk,
            self.start_index,
            self.progress,
        )?;
        if !continued {
            self.stopped = true;
        }
        Ok(continued)
    }

    fn consume_accounted(&mut self, chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
        <Self as qualified_accounted_transport::QualifiedSink>::consume_accounted_qualified(
            self, chunk,
        )
    }

    fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
        let qualified = qualified_accounted_chain(self.operators, self.sink);
        if qualified {
            Some(AccountedSinkPermit::new(self))
        } else {
            None
        }
    }

    fn admit_chunk_transport(&self, input: ChunkTransport) -> Result<(), OperatorError> {
        admit_chunk_transport_chain(self.operators, self.sink, input)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "StreamingPipelineSink"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        panic!("borrowed streaming pipeline sink cannot escape finalization")
    }
}

impl qualified_accounted_transport::QualifiedSink for StreamingPipelineSink<'_> {
    fn consume_accounted_qualified(
        &mut self,
        chunk: AccountedDataChunk,
    ) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        let continued = push_accounted_through_streaming_chain(
            self.operators,
            self.sink,
            self.checkpoint,
            chunk,
            self.start_index,
            self.progress,
        )?;
        if !continued {
            self.stopped = true;
        }
        Ok(continued)
    }
}

fn push_through_streaming_chain(
    operators: &mut [Box<dyn PushOperator>],
    sink: &mut dyn Sink,
    checkpoint: Option<&QueryExecutionCheckpoint>,
    chunk: DataChunk,
    start_index: usize,
    progress: &mut ChainProgress,
) -> Result<bool, OperatorError> {
    check_streaming_checkpoint(checkpoint)?;
    let continued = if let Some((operator, remaining)) = operators.split_first_mut() {
        let mut downstream = StreamingPipelineSink {
            operators: remaining,
            sink,
            checkpoint,
            stopped: false,
            start_index: start_index + 1,
            progress,
        };
        let operator_continued = operator.push(chunk, &mut downstream)?;
        if !operator_continued && !downstream.stopped {
            *downstream.progress = ChainProgress::ExhaustedAfter(start_index);
        }
        operator_continued && !downstream.stopped
    } else {
        let continued = sink.consume(chunk)?;
        if !continued {
            *progress = ChainProgress::SinkStopped;
        }
        continued
    };
    check_streaming_checkpoint(checkpoint)?;
    Ok(continued)
}

fn push_accounted_through_streaming_chain(
    operators: &mut [Box<dyn PushOperator>],
    sink: &mut dyn Sink,
    checkpoint: Option<&QueryExecutionCheckpoint>,
    chunk: AccountedDataChunk,
    start_index: usize,
    progress: &mut ChainProgress,
) -> Result<bool, OperatorError> {
    check_streaming_checkpoint(checkpoint)?;
    let continued = if let Some((operator, remaining)) = operators.split_first_mut() {
        let mut downstream = StreamingPipelineSink {
            operators: remaining,
            sink,
            checkpoint,
            stopped: false,
            start_index: start_index + 1,
            progress,
        };
        let operator_continued = {
            let mut downstream_permit = accounted_sink_permit(&mut downstream)?;
            accounted_push_permit(operator.as_mut())?.push(chunk, &mut downstream_permit)?
        };
        if !operator_continued && !downstream.stopped {
            *downstream.progress = ChainProgress::ExhaustedAfter(start_index);
        }
        operator_continued && !downstream.stopped
    } else {
        let continued = accounted_sink_permit(sink)?.consume(chunk)?;
        if !continued {
            *progress = ChainProgress::SinkStopped;
        }
        continued
    };
    check_streaming_checkpoint(checkpoint)?;
    Ok(continued)
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(
        source: Box<dyn Source>,
        operators: Vec<Box<dyn PushOperator>>,
        sink: Box<dyn Sink>,
    ) -> Self {
        Self {
            source,
            operators,
            sink,
            checkpoint: None,
            progress: ChainProgress::Continue,
        }
    }

    /// Create a simple pipeline with just source and sink.
    pub fn simple(source: Box<dyn Source>, sink: Box<dyn Sink>) -> Self {
        Self {
            source,
            operators: Vec::new(),
            sink,
            checkpoint: None,
            progress: ChainProgress::Continue,
        }
    }

    /// Add an operator to the pipeline.
    #[must_use]
    pub fn with_operator(mut self, op: Box<dyn PushOperator>) -> Self {
        self.operators.push(op);
        self
    }

    /// Validates public advisory transport declarations across the complete chain.
    ///
    /// The fold starts plain because [`Source`] produces compatibility chunks.
    /// Once any operator declares that it may emit an accounted envelope, the
    /// fold remains accounted for every later stage. This read-only check is
    /// useful for plan diagnostics, but public implementations cannot authorize
    /// production accounted delivery: execution separately performs mutable
    /// preflight and requires crate-sealed, exact-object permits before polling
    /// the source.
    ///
    /// # Errors
    ///
    /// Returns the first error reported by an operator or sink's public
    /// advisory declaration. A successful result is not sealed authorization.
    pub fn validate_chunk_transport(&self) -> Result<(), OperatorError> {
        admit_chunk_transport_chain(
            &self.operators,
            self.sink.as_ref(),
            ChunkTransport::PlainOnly,
        )
    }

    fn validate_chunk_transport_for_execution(&mut self) -> Result<(), OperatorError> {
        admit_qualified_chunk_transport_chain(
            &mut self.operators,
            self.sink.as_mut(),
            ChunkTransport::PlainOnly,
        )
    }

    /// Installs an orchestration checkpoint for cooperative cancellation.
    ///
    /// Compose deadlines on the execution control before installing its
    /// checkpoint. A terminal checkpoint remains terminal on repeated execution;
    /// a new execution requires a fresh checkpoint.
    ///
    /// ```
    /// use grafeo_core::execution::{QueryExecutionControl, pipeline::Pipeline};
    /// use grafeo_core::execution::{sink::NullSink, source::VectorSource};
    ///
    /// let control = QueryExecutionControl::new();
    /// let mut pipeline = Pipeline::simple(
    ///     Box::new(VectorSource::new(Vec::new())), Box::new(NullSink::new()),
    /// ).with_execution_checkpoint(control.checkpoint());
    /// pipeline.execute().unwrap();
    /// ```
    ///
    /// A raw deadline cannot create an independent control:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_core::execution::pipeline::Pipeline;
    /// fn unqualified(pipeline: Pipeline) {
    ///     let _ = pipeline.with_deadline(None);
    /// }
    /// ```
    #[must_use]
    pub fn with_execution_checkpoint(mut self, checkpoint: QueryExecutionCheckpoint) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Installs an orchestration checkpoint on an existing pipeline.
    ///
    /// A raw deadline cannot replace the execution's checkpoint:
    ///
    /// ```compile_fail,E0599
    /// use grafeo_core::execution::pipeline::Pipeline;
    /// fn unqualified(pipeline: &mut Pipeline) {
    ///     pipeline.set_deadline(None);
    /// }
    /// ```
    pub fn set_execution_checkpoint(&mut self, checkpoint: QueryExecutionCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    /// Checks the execution's shared cancellation and deadline state.
    fn check_cooperative_cancellation(&self) -> Result<(), OperatorError> {
        check_streaming_checkpoint(self.checkpoint.as_ref())
    }

    /// Consumes the pipeline and returns the sink.
    ///
    /// Call this after [`execute()`](Self::execute) to retrieve collected results
    /// from the sink. Useful for extracting chunks from a [`CollectorSink`](super::sink::CollectorSink)
    /// or [`ChunkCollector`].
    pub fn into_sink(self) -> Box<dyn Sink> {
        self.sink
    }

    /// Execute the pipeline, carrying [`ChunkVariant`] between operators.
    ///
    /// Operators that do not accept factorized chunks flatten at the boundary.
    /// Existing [`Self::execute`] stays flat-only.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any source, operator, or sink fails, or if the
    /// execution checkpoint reports cancellation or deadline expiry.
    pub fn execute_variant(&mut self) -> Result<(), OperatorError> {
        self.validate_chunk_transport_for_execution()?;
        self.progress = ChainProgress::Continue;
        let chunk_size = self.compute_chunk_size();

        loop {
            self.check_cooperative_cancellation()?;
            let next = self.source.next_chunk(chunk_size)?;
            let Some(chunk) = next else {
                self.check_cooperative_cancellation()?;
                break;
            };
            self.check_cooperative_cancellation()?;
            if !self.push_through_variant(ChunkVariant::Flat(chunk))? {
                break;
            }
        }

        self.check_cooperative_cancellation()?;
        self.finalize_all()?;
        self.check_cooperative_cancellation()
    }

    /// Execute the pipeline.
    ///
    /// # Deadline behavior
    ///
    /// With an explicit execution checkpoint, cancellation is checked before
    /// and after source calls, between push operators, and around finalization.
    /// A blocking syscall, opaque callback, or individual operator call remains
    /// cooperative rather than preemptive.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any source, operator, or sink fails during execution,
    /// or if the execution checkpoint reports cancellation or deadline expiry.
    pub fn execute(&mut self) -> Result<(), OperatorError> {
        self.validate_chunk_transport_for_execution()?;
        self.progress = ChainProgress::Continue;
        let chunk_size = self.compute_chunk_size();

        // Process all chunks from source
        loop {
            self.check_cooperative_cancellation()?;
            let next = self.source.next_chunk(chunk_size)?;
            let Some(chunk) = next else {
                self.check_cooperative_cancellation()?;
                break;
            };
            self.check_cooperative_cancellation()?;

            if !self.push_through(chunk)? {
                // Early termination requested
                break;
            }
        }

        // Finalize all operators (important for pipeline breakers)
        self.check_cooperative_cancellation()?;
        self.finalize_all()?;
        self.check_cooperative_cancellation()
    }

    /// Compute optimal chunk size from operator hints.
    fn compute_chunk_size(&self) -> usize {
        let mut size = DEFAULT_CHUNK_SIZE;

        for op in &self.operators {
            match op.preferred_chunk_size() {
                ChunkSizeHint::Default => {}
                ChunkSizeHint::Small => size = size.min(SMALL_CHUNK_SIZE),
                ChunkSizeHint::Large => size = size.max(LARGE_CHUNK_SIZE),
                ChunkSizeHint::Exact(s) => return s,
                ChunkSizeHint::AtMost(s) => size = size.min(s),
            }
        }

        size
    }

    /// Push a [`ChunkVariant`] through the operator chain.
    fn push_through_variant(&mut self, chunk: ChunkVariant) -> Result<bool, OperatorError> {
        self.check_cooperative_cancellation()?;
        if self.operators.is_empty() {
            let result = self.sink.consume_variant(chunk)?;
            self.check_cooperative_cancellation()?;
            if !result {
                self.progress = ChainProgress::SinkStopped;
            }
            return Ok(result);
        }

        enum PendingVariant {
            Chunk(usize, ChunkVariant),
            Stop(usize),
        }

        let num_operators = self.operators.len();
        let mut pending = vec![PendingVariant::Chunk(0, chunk)];
        while let Some(item) = pending.pop() {
            self.check_cooperative_cancellation()?;
            let (start, current) = match item {
                PendingVariant::Chunk(start, chunk) => (start, chunk),
                PendingVariant::Stop(index) => {
                    self.progress = ChainProgress::ExhaustedAfter(index);
                    return Ok(false);
                }
            };
            if start >= num_operators {
                continue;
            }

            if start == num_operators - 1 {
                let (continued, terminal_stopped) = {
                    let mut terminal = TerminalContinuationSink {
                        sink: &mut *self.sink,
                        stopped: false,
                    };
                    let operator_continued =
                        self.operators[start].push_variant(current, &mut terminal)?;
                    (operator_continued && !terminal.stopped, terminal.stopped)
                };
                self.check_cooperative_cancellation()?;
                if !continued {
                    self.progress = if terminal_stopped {
                        ChainProgress::SinkStopped
                    } else {
                        ChainProgress::ExhaustedAfter(start)
                    };
                    return Ok(false);
                }
                continue;
            }

            let mut collector = VariantCollector::new();
            let continue_processing =
                self.operators[start].push_variant(current, &mut collector)?;
            self.check_cooperative_cancellation()?;
            let chunks = collector.into_chunks();
            if !continue_processing {
                pending.push(PendingVariant::Stop(start));
            }
            for next in chunks.into_iter().rev() {
                pending.push(PendingVariant::Chunk(start + 1, next));
            }
        }

        Ok(true)
    }

    /// Push a chunk through the operator chain.
    fn push_through(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.check_cooperative_cancellation()?;
        if self.operators.is_empty() {
            // No operators, push directly to sink
            let result = self.sink.consume(chunk)?;
            self.check_cooperative_cancellation()?;
            if !result {
                self.progress = ChainProgress::SinkStopped;
            }
            return Ok(result);
        }

        enum PendingChunk {
            Chunk(usize, DataChunk),
            Stop(usize),
        }

        let num_operators = self.operators.len();
        let mut pending = vec![PendingChunk::Chunk(0, chunk)];
        while let Some(item) = pending.pop() {
            self.check_cooperative_cancellation()?;
            let (start, current_chunk) = match item {
                PendingChunk::Chunk(start, chunk) => (start, chunk),
                PendingChunk::Stop(index) => {
                    self.progress = ChainProgress::ExhaustedAfter(index);
                    return Ok(false);
                }
            };
            if start >= num_operators {
                continue;
            }

            if start == num_operators - 1 {
                // Last operator pushes through a continuation guard so it
                // cannot swallow the real sink's terminal `false` response.
                let (continued, terminal_stopped) = {
                    let mut terminal = TerminalContinuationSink {
                        sink: &mut *self.sink,
                        stopped: false,
                    };
                    let operator_continued =
                        self.operators[start].push(current_chunk, &mut terminal)?;
                    (operator_continued && !terminal.stopped, terminal.stopped)
                };
                self.check_cooperative_cancellation()?;
                if !continued {
                    self.progress = if terminal_stopped {
                        ChainProgress::SinkStopped
                    } else {
                        ChainProgress::ExhaustedAfter(start)
                    };
                    return Ok(false);
                }
                continue;
            }

            // Keep intermediate output chunks separate. A single DataChunk
            // cannot represent mixed row schemas.
            let mut collector = ChunkCollector::new();
            let continue_processing = self.operators[start].push(current_chunk, &mut collector)?;
            self.check_cooperative_cancellation()?;

            let chunks = collector.into_chunks();
            if !continue_processing {
                pending.push(PendingChunk::Stop(start));
            }
            for next in chunks.into_iter().rev() {
                pending.push(PendingChunk::Chunk(start + 1, next));
            }
        }

        Ok(true)
    }

    /// Finalize only the suffix still entitled to emit accepted rows.
    fn finalize_all(&mut self) -> Result<(), OperatorError> {
        self.check_cooperative_cancellation()?;
        finalize_operator_chain(
            &mut self.operators,
            &mut *self.sink,
            self.checkpoint.as_ref(),
            self.progress,
        )?;
        self.check_cooperative_cancellation()
    }
}

/// Drains accepted downstream state after producer exhaustion. Terminal sink
/// stop forbids later operator emissions, while sink finalization still runs.
pub(super) fn finalize_operator_chain(
    operators: &mut [Box<dyn PushOperator>],
    sink: &mut dyn Sink,
    checkpoint: Option<&QueryExecutionCheckpoint>,
    progress: ChainProgress,
) -> Result<(), OperatorError> {
    let mut next = progress.finalize_start(operators.len());
    while next < operators.len() {
        check_streaming_checkpoint(checkpoint)?;
        let mut emitted_progress = ChainProgress::Continue;
        let (through_current, remaining) = operators.split_at_mut(next + 1);
        let mut downstream = StreamingPipelineSink {
            operators: remaining,
            sink,
            checkpoint,
            stopped: false,
            start_index: next + 1,
            progress: &mut emitted_progress,
        };
        through_current[next].finalize(&mut downstream)?;
        check_streaming_checkpoint(checkpoint)?;
        next = if emitted_progress.continued() {
            next + 1
        } else {
            emitted_progress.finalize_start(operators.len())
        };
    }
    check_streaming_checkpoint(checkpoint)?;
    sink.finalize()?;
    check_streaming_checkpoint(checkpoint)
}

/// Collects flat or factorized chunks between variant-aware operators.
pub struct VariantCollector {
    chunks: Vec<ChunkVariant>,
}

impl VariantCollector {
    /// Create an empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// True when nothing was collected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Returns collected variants without merging their flat payloads.
    #[must_use]
    pub fn into_chunks(self) -> Vec<ChunkVariant> {
        self.chunks
    }

    /// Merge collected variants. Mixed flat/factorized flattens.
    #[must_use]
    pub fn into_single_variant(self) -> ChunkVariant {
        if self.chunks.len() == 1 {
            return self
                .chunks
                .into_iter()
                .next()
                .unwrap_or_else(|| ChunkVariant::Flat(DataChunk::empty()));
        }
        let flats: Vec<DataChunk> = self
            .chunks
            .into_iter()
            .map(ChunkVariant::ensure_flat)
            .collect();
        if flats.is_empty() {
            return ChunkVariant::Flat(DataChunk::empty());
        }
        ChunkVariant::Flat(DataChunk::concat(&flats))
    }
}

impl Default for VariantCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for VariantCollector {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.chunks.push(ChunkVariant::Flat(chunk));
        Ok(true)
    }

    fn consume_variant(&mut self, chunk: ChunkVariant) -> Result<bool, OperatorError> {
        self.chunks.push(chunk);
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "VariantCollector"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// A terminal consumer that borrows each input only for the duration of a call.
///
/// Implementations must admit and retain their own memory authority before
/// copying input into result storage. This interface never transfers the input
/// envelope's grant, even when the pipeline delivers accounted data.
pub trait ResultConsumer: Send + Sync + 'static {
    /// Consumes a borrowed flat input under the consumer's output budget.
    ///
    /// # Errors
    /// Returns output admission or conversion failures before retaining data.
    fn consume(&mut self, chunk: &DataChunk) -> Result<bool, OperatorError>;

    /// Consumes a borrowed factorized input without an unbounded flatten step.
    ///
    /// # Errors
    /// Returns output admission or conversion failures before retaining data.
    fn consume_variant(&mut self, chunk: &ChunkVariant) -> Result<bool, OperatorError>;

    /// Finalizes the bounded consumer without emitting additional input.
    ///
    /// # Errors
    /// Returns mandatory consumer finalization failures.
    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }
}

/// A terminal bridge that keeps each input's authority alive until consumption.
///
/// Accounted chunks are never unwrapped or forwarded to an arbitrary public
/// `Sink` implementation. The bridge lends their immutable view, retaining the
/// complete envelope throughout the bounded consumer's call and any unwind.
pub struct ResultConsumerSink<T: ResultConsumer> {
    consumer: T,
}

impl<T: ResultConsumer> ResultConsumerSink<T> {
    /// Creates a terminal sink around an output-budgeted consumer.
    pub fn new(consumer: T) -> Self {
        Self { consumer }
    }

    /// Returns the consumer and its retained output authority after execution.
    pub fn into_inner(self) -> T {
        self.consumer
    }
}

impl<T: ResultConsumer> Sink for ResultConsumerSink<T> {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.consumer.consume(&chunk)
    }

    fn consume_variant(&mut self, chunk: ChunkVariant) -> Result<bool, OperatorError> {
        self.consumer.consume_variant(&chunk)
    }

    fn consume_accounted(&mut self, chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
        self.consumer.consume(chunk.chunk())
    }

    fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
        Some(AccountedSinkPermit::new(self))
    }

    fn admit_chunk_transport(&self, _input: ChunkTransport) -> Result<(), OperatorError> {
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        self.consumer.finalize()
    }

    fn name(&self) -> &'static str {
        "ResultConsumerSink"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

impl<T: ResultConsumer> qualified_accounted_transport::QualifiedSink for ResultConsumerSink<T> {
    fn consume_accounted_qualified(
        &mut self,
        chunk: AccountedDataChunk,
    ) -> Result<bool, OperatorError> {
        self.consumer.consume(chunk.chunk())
    }
}

/// Collects chunks from operators for intermediate processing.
pub struct ChunkCollector {
    chunks: Vec<DataChunk>,
}

impl ChunkCollector {
    /// Create a new chunk collector.
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Check if collector has any chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get total row count across all chunks.
    pub fn row_count(&self) -> usize {
        self.chunks.iter().map(DataChunk::len).sum()
    }

    /// Convert to a vector of chunks.
    pub fn into_chunks(self) -> Vec<DataChunk> {
        self.chunks
    }

    /// Merge all chunks into a single chunk.
    ///
    /// # Panics
    ///
    /// Panics if internal invariants are violated (single-element vec is unexpectedly empty).
    pub fn into_single_chunk(self) -> DataChunk {
        if self.chunks.is_empty() {
            return DataChunk::empty();
        }
        if self.chunks.len() == 1 {
            // Invariant: self.chunks.len() == 1 guarantees exactly one element
            return self
                .chunks
                .into_iter()
                .next()
                .expect("chunks has exactly one element: checked on previous line");
        }

        // Concatenate all chunks
        DataChunk::concat(&self.chunks)
    }
}

impl Default for ChunkCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for ChunkCollector {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if !chunk.is_empty() {
            self.chunks.push(chunk);
        }
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChunkCollector"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::push::{SortKey, SortPushOperator};
    use crate::execution::vector::ValueVector;
    use crate::execution::{
        QueryCancellationError, QueryCancellationHandle, QueryExecutionControl,
        QueryResourceContext,
    };
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::{LogicalType, Value};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test source that produces a fixed number of chunks.
    struct TestSource {
        remaining: usize,
        values_per_chunk: usize,
    }

    impl TestSource {
        fn new(num_chunks: usize, values_per_chunk: usize) -> Self {
            Self {
                remaining: num_chunks,
                values_per_chunk,
            }
        }
    }

    impl Source for TestSource {
        fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;

            // Create a chunk with integer values
            // reason: test chunk size is small, fits i64
            #[allow(clippy::cast_possible_wrap)]
            let values: Vec<Value> = (0..self.values_per_chunk)
                .map(|i| Value::Int64(i as i64))
                .collect();
            let vector = ValueVector::from_values(&values);
            let chunk = DataChunk::new(vec![vector]);
            Ok(Some(chunk))
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "TestSource"
        }
    }

    /// Test sink that collects all chunks.
    struct TestSink {
        chunks: Vec<DataChunk>,
        finalized: bool,
    }

    impl TestSink {
        fn new() -> Self {
            Self {
                chunks: Vec::new(),
                finalized: false,
            }
        }
    }

    impl Sink for TestSink {
        fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
            self.chunks.push(chunk);
            Ok(true)
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            self.finalized = true;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "TestSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    /// Pass-through operator for testing.
    struct PassThroughOperator;

    impl PushOperator for PassThroughOperator {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "PassThrough"
        }
    }

    fn provenance_edge_chunk(id: i64) -> DataChunk {
        let mut column = ValueVector::with_type(LogicalType::List(Box::new(LogicalType::Edge)));
        column.push(Value::List(vec![Value::Int64(id)].into()));
        DataChunk::new(vec![column])
    }

    fn provenance_any_chunk(id: i64) -> DataChunk {
        let mut column = ValueVector::new();
        column.push(Value::List(vec![Value::Int64(id)].into()));
        DataChunk::new(vec![column])
    }

    struct MixedProvenanceEmitter;

    impl PushOperator for MixedProvenanceEmitter {
        fn push(&mut self, _chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(provenance_edge_chunk(1))?;
            sink.consume(provenance_any_chunk(2))?;
            Ok(true)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "MixedProvenanceEmitter"
        }
    }

    struct ProvenanceRecorder {
        observed: Arc<Mutex<Vec<LogicalType>>>,
    }

    impl PushOperator for ProvenanceRecorder {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            if let Some(column) = chunk.column(0) {
                self.observed
                    .lock()
                    .expect("provenance recorder lock")
                    .push(column.data_type().clone());
            }
            sink.consume(chunk)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "ProvenanceRecorder"
        }
    }

    struct DropTypedEdgeRun;

    impl PushOperator for DropTypedEdgeRun {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            let is_typed_edge = chunk.column(0).is_some_and(|column| {
                matches!(
                    column.data_type(),
                    LogicalType::List(item) if item.as_ref() == &LogicalType::Edge
                )
            });
            if is_typed_edge {
                Ok(true)
            } else {
                sink.consume(chunk)
            }
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "DropTypedEdgeRun"
        }
    }

    fn run_mixed_provenance_pipeline(variant: bool) -> Vec<LogicalType> {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline =
            Pipeline::simple(Box::new(TestSource::new(1, 1)), Box::new(TestSink::new()))
                .with_operator(Box::new(MixedProvenanceEmitter))
                .with_operator(Box::new(ProvenanceRecorder {
                    observed: Arc::clone(&observed),
                }));
        if variant {
            pipeline.execute_variant().unwrap();
        } else {
            pipeline.execute().unwrap();
        }
        drop(pipeline);
        Arc::try_unwrap(observed)
            .expect("provenance recorder has no remaining references")
            .into_inner()
            .expect("provenance recorder lock")
    }

    #[test]
    fn intermediate_pipeline_preserves_mixed_provenance_runs() {
        let expected = vec![
            LogicalType::List(Box::new(LogicalType::Edge)),
            LogicalType::Any,
        ];
        assert_eq!(run_mixed_provenance_pipeline(false), expected);
        assert_eq!(run_mixed_provenance_pipeline(true), expected);
    }

    #[test]
    fn intermediate_empty_run_does_not_discard_later_schema_run() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline =
            Pipeline::simple(Box::new(TestSource::new(1, 1)), Box::new(TestSink::new()))
                .with_operator(Box::new(MixedProvenanceEmitter))
                .with_operator(Box::new(DropTypedEdgeRun))
                .with_operator(Box::new(ProvenanceRecorder {
                    observed: Arc::clone(&observed),
                }));

        pipeline.execute().unwrap();

        assert_eq!(*observed.lock().unwrap(), vec![LogicalType::Any]);
    }

    #[test]
    fn intermediate_limit_forwards_buffered_output_before_stop() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline =
            Pipeline::simple(Box::new(TestSource::new(1, 3)), Box::new(TestSink::new()))
                .with_operator(Box::new(
                    crate::execution::operators::push::LimitPushOperator::new(1),
                ))
                .with_operator(Box::new(ProvenanceRecorder {
                    observed: Arc::clone(&observed),
                }));

        pipeline.execute().unwrap();

        assert_eq!(*observed.lock().unwrap(), vec![LogicalType::Any]);
    }

    fn count_push_operator() -> Box<dyn PushOperator> {
        Box::new(
            crate::execution::operators::push::AggregatePushOperator::new(
                Vec::new(),
                vec![crate::execution::operators::accumulator::AggregateExpr::count_star()],
            ),
        )
    }

    fn assert_pipeline_count(mut pipeline: Pipeline, variant: bool, expected: i64) {
        if variant {
            pipeline.execute_variant().unwrap();
        } else {
            pipeline.execute().unwrap();
        }
        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<TestSink>()
            .unwrap();
        let actual: Vec<_> = sink
            .chunks
            .iter()
            .flat_map(|chunk| {
                chunk
                    .selected_indices()
                    .map(|row| chunk.column(0).unwrap().get_value(row).unwrap())
            })
            .collect();
        assert_eq!(actual, vec![Value::Int64(expected)]);
    }

    #[test]
    fn limit_exhaustion_finalizes_downstream_count() {
        for variant in [false, true] {
            let pipeline = Pipeline::new(
                Box::new(TestSource::new(2, 3)),
                vec![
                    Box::new(crate::execution::operators::push::LimitPushOperator::new(1)),
                    count_push_operator(),
                ],
                Box::new(TestSink::new()),
            );
            assert_pipeline_count(pipeline, variant, 1);
        }
    }

    #[test]
    fn sorter_limit_exhaustion_finalizes_downstream_count() {
        for variant in [false, true] {
            let resources =
                QueryResourceContext::new(BufferManager::with_budget(1024 * 1024)).unwrap();
            let pipeline = Pipeline::new(
                Box::new(TestSource::new(2, 3)),
                vec![
                    Box::new(
                        SortPushOperator::with_resource_context(
                            vec![SortKey::ascending(0)],
                            resources,
                        )
                        .unwrap(),
                    ),
                    Box::new(crate::execution::operators::push::LimitPushOperator::new(1)),
                    count_push_operator(),
                ],
                Box::new(TestSink::new()),
            );
            assert_pipeline_count(pipeline, variant, 1);
        }
    }

    #[test]
    fn exhausted_producer_is_not_finalized_but_downstream_count_is() {
        for variant in [false, true] {
            let finalizes = Arc::new(AtomicUsize::new(0));
            let pipeline = Pipeline::new(
                Box::new(TestSource::new(2, 1)),
                vec![
                    Box::new(ContinuationProbeOperator {
                        pushes: Arc::new(AtomicUsize::new(0)),
                        finalizes: finalizes.clone(),
                        forward: true,
                        continue_after_push: false,
                        fail_on_finalize: true,
                    }),
                    count_push_operator(),
                ],
                Box::new(TestSink::new()),
            );
            assert_pipeline_count(pipeline, variant, 1);
            assert_eq!(finalizes.load(Ordering::Acquire), 0);
        }
    }

    struct FinalizeEmitter;

    impl PushOperator for FinalizeEmitter {
        fn push(&mut self, _chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            Ok(true)
        }

        fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
            for value in [1, 2] {
                let chunk = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(value)])]);
                let _ = sink.consume(chunk)?;
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "FinalizeEmitter"
        }
    }

    struct PushEmitterIgnoringContinuation {
        finalizes: Arc<AtomicUsize>,
    }

    impl PushOperator for PushEmitterIgnoringContinuation {
        fn push(&mut self, _chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            for value in [1, 2] {
                let chunk = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(value)])]);
                let _ = sink.consume(chunk)?;
            }
            Ok(true)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            self.finalizes.fetch_add(1, Ordering::AcqRel);
            Err(OperatorError::Execution(
                "operator was finalized after terminal push stop".to_string(),
            ))
        }

        fn name(&self) -> &'static str {
            "PushEmitterIgnoringContinuation"
        }
    }

    struct ContinuationProbeOperator {
        pushes: Arc<AtomicUsize>,
        finalizes: Arc<AtomicUsize>,
        forward: bool,
        continue_after_push: bool,
        fail_on_finalize: bool,
    }

    impl PushOperator for ContinuationProbeOperator {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            self.pushes.fetch_add(1, Ordering::AcqRel);
            let downstream_continues = if self.forward {
                sink.consume(chunk)?
            } else {
                false
            };
            Ok(self.continue_after_push && (!self.forward || downstream_continues))
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            self.finalizes.fetch_add(1, Ordering::AcqRel);
            if self.fail_on_finalize {
                return Err(OperatorError::Execution(
                    "stopped operator was finalized".to_string(),
                ));
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "ContinuationProbeOperator"
        }
    }

    struct ContinuationProbeSink {
        consumes: Arc<AtomicUsize>,
        finalizes: Arc<AtomicUsize>,
        continue_consuming: bool,
    }

    impl Sink for ContinuationProbeSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            let previous = self.consumes.fetch_add(1, Ordering::AcqRel);
            assert!(
                self.continue_consuming || previous == 0,
                "sink was called again after returning false"
            );
            Ok(self.continue_consuming)
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            self.finalizes.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn name(&self) -> &'static str {
            "ContinuationProbeSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    struct CancellingSource {
        cancellation: QueryCancellationHandle,
        fail: bool,
        emit_chunk: bool,
    }

    impl Source for CancellingSource {
        fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
            self.cancellation.cancel();
            if self.fail {
                Err(OperatorError::Execution("source failed".to_string()))
            } else if self.emit_chunk {
                Ok(Some(DataChunk::new(vec![ValueVector::from_values(&[
                    Value::Int64(7),
                ])])))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "CancellingSource"
        }
    }

    struct ObservedSource {
        source: TestSource,
        polls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl Source for ObservedSource {
        fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
            self.polls.fetch_add(1, Ordering::AcqRel);
            if self.fail {
                return Err(OperatorError::Execution("source failed".into()));
            }
            self.source.next_chunk(chunk_size)
        }

        fn reset(&mut self) {
            self.source.reset();
        }

        fn name(&self) -> &'static str {
            "ObservedSource"
        }
    }

    fn execute_test_pipeline(pipeline: &mut Pipeline, variant: bool) -> Result<(), OperatorError> {
        if variant {
            pipeline.execute_variant()
        } else {
            pipeline.execute()
        }
    }

    struct CancellingPushOperator {
        cancellation: QueryCancellationHandle,
        cancel_on_push: bool,
        cancel_on_finalize: bool,
        fail: bool,
    }

    impl PushOperator for CancellingPushOperator {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            if self.cancel_on_push {
                self.cancellation.cancel();
                if self.fail {
                    return Err(OperatorError::Execution("push failed".to_string()));
                }
            }
            sink.consume(chunk)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            if self.cancel_on_finalize {
                self.cancellation.cancel();
                if self.fail {
                    return Err(OperatorError::Execution(
                        "operator finalize failed".to_string(),
                    ));
                }
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "CancellingPush"
        }
    }

    struct CancellingSink {
        cancellation: QueryCancellationHandle,
        cancel_on_consume: bool,
        cancel_on_finalize: bool,
        fail: bool,
    }

    impl Sink for CancellingSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            if self.cancel_on_consume {
                self.cancellation.cancel();
                if self.fail {
                    return Err(OperatorError::Execution("sink consume failed".to_string()));
                }
            }
            Ok(true)
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            if self.cancel_on_finalize {
                self.cancellation.cancel();
                if self.fail {
                    return Err(OperatorError::Execution("sink finalize failed".to_string()));
                }
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "CancellingSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    #[test]
    fn test_simple_pipeline() {
        let source = Box::new(TestSource::new(3, 10));
        let sink = Box::new(TestSink::new());

        let mut pipeline = Pipeline::simple(source, sink);
        pipeline.execute().unwrap();

        // Access sink through downcast (in real code we'd use a different pattern)
        // For this test, we verify execution completed without error
    }

    #[test]
    fn test_pipeline_with_operator() {
        let source = Box::new(TestSource::new(2, 5));
        let sink = Box::new(TestSink::new());

        let mut pipeline =
            Pipeline::simple(source, sink).with_operator(Box::new(PassThroughOperator));

        pipeline.execute().unwrap();
    }

    #[test]
    fn finalize_output_reaches_downstream_before_upstream_finalize_returns() {
        struct FinalizingProducer {
            downstream_observed: Arc<AtomicBool>,
        }

        impl PushOperator for FinalizingProducer {
            fn push(
                &mut self,
                _chunk: DataChunk,
                _sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                Ok(true)
            }

            fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
                let chunk = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(1)])]);
                sink.consume(chunk)?;
                if !self.downstream_observed.load(Ordering::Acquire) {
                    return Err(OperatorError::Execution(
                        "pipeline buffered finalize output instead of streaming it".to_string(),
                    ));
                }
                let chunk = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(2)])]);
                sink.consume(chunk)?;
                Ok(())
            }

            fn name(&self) -> &'static str {
                "FinalizingProducer"
            }
        }

        struct ObservingPassThrough {
            observed: Arc<AtomicBool>,
        }

        impl PushOperator for ObservingPassThrough {
            fn push(
                &mut self,
                chunk: DataChunk,
                sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                self.observed.store(true, Ordering::Release);
                sink.consume(chunk)
            }

            fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "ObservingPassThrough"
            }
        }

        let observed = Arc::new(AtomicBool::new(false));
        let operators: Vec<Box<dyn PushOperator>> = vec![
            Box::new(FinalizingProducer {
                downstream_observed: Arc::clone(&observed),
            }),
            Box::new(ObservingPassThrough {
                observed: Arc::clone(&observed),
            }),
        ];
        let mut pipeline = Pipeline::new(
            Box::new(TestSource::new(0, 1)),
            operators,
            Box::new(TestSink::new()),
        );

        pipeline.execute().unwrap();
        assert!(observed.load(Ordering::Acquire));
    }

    #[test]
    fn finalize_exhaustion_skips_producer_but_finalizes_downstream() {
        let stop_pushes = Arc::new(AtomicUsize::new(0));
        let stop_finalizes = Arc::new(AtomicUsize::new(0));
        let later_pushes = Arc::new(AtomicUsize::new(0));
        let later_finalizes = Arc::new(AtomicUsize::new(0));
        let sink_consumes = Arc::new(AtomicUsize::new(0));
        let sink_finalizes = Arc::new(AtomicUsize::new(0));
        let operators: Vec<Box<dyn PushOperator>> = vec![
            Box::new(FinalizeEmitter),
            Box::new(ContinuationProbeOperator {
                pushes: Arc::clone(&stop_pushes),
                finalizes: Arc::clone(&stop_finalizes),
                forward: false,
                continue_after_push: false,
                fail_on_finalize: true,
            }),
            Box::new(ContinuationProbeOperator {
                pushes: Arc::clone(&later_pushes),
                finalizes: Arc::clone(&later_finalizes),
                forward: true,
                continue_after_push: true,
                fail_on_finalize: false,
            }),
        ];
        let mut pipeline = Pipeline::new(
            Box::new(TestSource::new(0, 1)),
            operators,
            Box::new(ContinuationProbeSink {
                consumes: Arc::clone(&sink_consumes),
                finalizes: Arc::clone(&sink_finalizes),
                continue_consuming: true,
            }),
        );

        pipeline.execute().unwrap();
        assert_eq!(stop_pushes.load(Ordering::Acquire), 1);
        assert_eq!(stop_finalizes.load(Ordering::Acquire), 0);
        assert_eq!(later_pushes.load(Ordering::Acquire), 0);
        assert_eq!(later_finalizes.load(Ordering::Acquire), 1);
        assert_eq!(sink_consumes.load(Ordering::Acquire), 0);
        assert_eq!(sink_finalizes.load(Ordering::Acquire), 1);
    }

    #[test]
    fn push_stop_skips_operator_finalization_but_finalizes_sink() {
        for variant_execution in [false, true] {
            let operator_pushes = Arc::new(AtomicUsize::new(0));
            let operator_finalizes = Arc::new(AtomicUsize::new(0));
            let sink_consumes = Arc::new(AtomicUsize::new(0));
            let sink_finalizes = Arc::new(AtomicUsize::new(0));
            let mut pipeline = Pipeline::simple(
                Box::new(TestSource::new(1, 1)),
                Box::new(ContinuationProbeSink {
                    consumes: Arc::clone(&sink_consumes),
                    finalizes: Arc::clone(&sink_finalizes),
                    continue_consuming: true,
                }),
            )
            .with_operator(Box::new(ContinuationProbeOperator {
                pushes: Arc::clone(&operator_pushes),
                finalizes: Arc::clone(&operator_finalizes),
                forward: false,
                continue_after_push: false,
                fail_on_finalize: true,
            }));

            if variant_execution {
                pipeline.execute_variant().unwrap();
            } else {
                pipeline.execute().unwrap();
            }
            assert_eq!(operator_pushes.load(Ordering::Acquire), 1);
            assert_eq!(operator_finalizes.load(Ordering::Acquire), 0);
            assert_eq!(sink_consumes.load(Ordering::Acquire), 0);
            assert_eq!(sink_finalizes.load(Ordering::Acquire), 1);
        }
    }

    #[test]
    fn terminal_sink_stop_blocks_repeated_finalize_emission() {
        let consumes = Arc::new(AtomicUsize::new(0));
        let finalizes = Arc::new(AtomicUsize::new(0));
        let mut pipeline = Pipeline::simple(
            Box::new(TestSource::new(0, 1)),
            Box::new(ContinuationProbeSink {
                consumes: Arc::clone(&consumes),
                finalizes: Arc::clone(&finalizes),
                continue_consuming: false,
            }),
        )
        .with_operator(Box::new(FinalizeEmitter));

        pipeline.execute().unwrap();
        assert_eq!(consumes.load(Ordering::Acquire), 1);
        assert_eq!(finalizes.load(Ordering::Acquire), 1);
    }

    #[test]
    fn terminal_sink_stop_cannot_be_swallowed_during_push() {
        for variant_execution in [false, true] {
            let operator_finalizes = Arc::new(AtomicUsize::new(0));
            let sink_consumes = Arc::new(AtomicUsize::new(0));
            let sink_finalizes = Arc::new(AtomicUsize::new(0));
            let mut pipeline = Pipeline::simple(
                Box::new(TestSource::new(1, 1)),
                Box::new(ContinuationProbeSink {
                    consumes: Arc::clone(&sink_consumes),
                    finalizes: Arc::clone(&sink_finalizes),
                    continue_consuming: false,
                }),
            )
            .with_operator(Box::new(PushEmitterIgnoringContinuation {
                finalizes: Arc::clone(&operator_finalizes),
            }));

            if variant_execution {
                pipeline.execute_variant().unwrap();
            } else {
                pipeline.execute().unwrap();
            }
            assert_eq!(sink_consumes.load(Ordering::Acquire), 1);
            assert_eq!(sink_finalizes.load(Ordering::Acquire), 1);
            assert_eq!(operator_finalizes.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn terminal_continuation_guard_preserves_factorized_chunks() {
        use crate::execution::factorized_chunk::FactorizedChunk;
        use grafeo_common::types::LogicalType;

        let mut values = ValueVector::with_type(LogicalType::Int64);
        values.push_int64(1);
        values.push_int64(2);
        let chunk = ChunkVariant::factorized(FactorizedChunk::with_flat_level(
            vec![values],
            vec!["value".into()],
        ));
        let mut collector = VariantCollector::new();
        {
            let mut terminal = TerminalContinuationSink {
                sink: &mut collector,
                stopped: false,
            };
            assert!(terminal.consume_variant(chunk).unwrap());
            assert!(!terminal.stopped);
        }

        let output = collector.into_single_variant();
        assert!(output.is_factorized());
        assert_eq!(output.logical_row_count(), 2);
    }

    #[test]
    fn test_chunk_collector() {
        let mut collector = ChunkCollector::new();
        assert!(collector.is_empty());

        let values: Vec<Value> = vec![Value::Int64(1), Value::Int64(2)];
        let vector = ValueVector::from_values(&values);
        let chunk = DataChunk::new(vec![vector]);

        collector.consume(chunk).unwrap();
        assert!(!collector.is_empty());
        assert_eq!(collector.row_count(), 2);

        let merged = collector.into_single_chunk();
        assert_eq!(merged.len(), 2);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_pipeline_deadline_expired() {
        use std::time::Duration;

        for variant in [false, true] {
            let control = QueryExecutionControl::with_timeout(Duration::ZERO).unwrap();
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(10, 5)), Box::new(TestSink::new()))
                    .with_execution_checkpoint(control.checkpoint());
            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::DeadlineExceeded {
                        timeout: Some(Duration::ZERO),
                    }
                ))
            ));
        }
    }

    #[test]
    fn test_pipeline_no_deadline() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(3, 5)), Box::new(TestSink::new()))
                    .with_execution_checkpoint(control.checkpoint());
            execute_test_pipeline(&mut pipeline, variant).unwrap();
            let sink = pipeline
                .into_sink()
                .into_any()
                .downcast::<TestSink>()
                .unwrap();
            assert_eq!(sink.chunks.iter().map(DataChunk::len).sum::<usize>(), 15);
            assert!(sink.finalized);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_pipeline_set_execution_checkpoint() {
        use std::time::Duration;

        for variant in [false, true] {
            let control = QueryExecutionControl::with_timeout(Duration::ZERO).unwrap();
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(10, 5)), Box::new(TestSink::new()));
            pipeline.set_execution_checkpoint(control.checkpoint());
            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::DeadlineExceeded {
                        timeout: Some(Duration::ZERO),
                    }
                ))
            ));
        }
    }

    #[test]
    fn pipeline_observes_explicit_cancellation_before_source_work() {
        for variant in [false, true] {
            let control = super::super::QueryExecutionControl::new();
            let checkpoint = control.checkpoint();
            assert!(control.cancellation_handle().try_cancel());
            let source = Box::new(TestSource::new(10, 5));
            let sink = Box::new(TestSink::new());
            let mut pipeline = Pipeline::simple(source, sink).with_execution_checkpoint(checkpoint);

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    super::super::QueryCancellationError::Cancelled
                ))
            ));
        }
    }

    #[test]
    fn concrete_source_error_beats_racing_cancellation() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let source = Box::new(CancellingSource {
                cancellation: control.cancellation_handle(),
                fail: true,
                emit_chunk: false,
            });
            let mut pipeline = Pipeline::simple(source, Box::new(TestSink::new()))
                .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::Execution(ref message)) if message == "source failed"
            ));
        }
    }

    #[test]
    fn concrete_push_error_beats_racing_cancellation() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let operator = CancellingPushOperator {
                cancellation: control.cancellation_handle(),
                cancel_on_push: true,
                cancel_on_finalize: false,
                fail: true,
            };
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(1, 1)), Box::new(TestSink::new()))
                    .with_operator(Box::new(operator))
                    .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::Execution(ref message)) if message == "push failed"
            ));
        }
    }

    #[test]
    fn concrete_sink_consume_error_beats_racing_cancellation() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let sink = CancellingSink {
                cancellation: control.cancellation_handle(),
                cancel_on_consume: true,
                cancel_on_finalize: false,
                fail: true,
            };
            let mut pipeline = Pipeline::simple(Box::new(TestSource::new(1, 1)), Box::new(sink))
                .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::Execution(ref message)) if message == "sink consume failed"
            ));
        }
    }

    #[test]
    fn concrete_operator_finalize_error_beats_racing_cancellation() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let operator = CancellingPushOperator {
                cancellation: control.cancellation_handle(),
                cancel_on_push: false,
                cancel_on_finalize: true,
                fail: true,
            };
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(0, 1)), Box::new(TestSink::new()))
                    .with_operator(Box::new(operator))
                    .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::Execution(ref message))
                    if message == "operator finalize failed"
            ));
        }
    }

    #[test]
    fn concrete_sink_finalize_error_beats_racing_cancellation() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let sink = CancellingSink {
                cancellation: control.cancellation_handle(),
                cancel_on_consume: false,
                cancel_on_finalize: true,
                fail: true,
            };
            let mut pipeline = Pipeline::simple(Box::new(TestSource::new(0, 1)), Box::new(sink))
                .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::Execution(ref message)) if message == "sink finalize failed"
            ));
        }
    }

    #[test]
    fn cancellation_after_final_empty_source_pull_is_observed() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let source = Box::new(CancellingSource {
                cancellation: control.cancellation_handle(),
                fail: false,
                emit_chunk: false,
            });
            let mut pipeline = Pipeline::simple(source, Box::new(TestSink::new()))
                .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::Cancelled
                ))
            ));
        }
    }

    #[test]
    fn cancellation_after_successful_sink_finalize_is_observed() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let sink = CancellingSink {
                cancellation: control.cancellation_handle(),
                cancel_on_consume: false,
                cancel_on_finalize: true,
                fail: false,
            };
            let mut pipeline = Pipeline::simple(Box::new(TestSource::new(0, 1)), Box::new(sink))
                .with_execution_checkpoint(control.checkpoint());

            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::Cancelled
                ))
            ));
        }
    }

    #[test]
    fn cancellation_after_successful_source_chunk_prevents_consumption() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let source = CancellingSource {
                cancellation: control.cancellation_handle(),
                fail: false,
                emit_chunk: true,
            };
            let mut pipeline = Pipeline::simple(Box::new(source), Box::new(TestSink::new()))
                .with_execution_checkpoint(control.checkpoint());
            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::Cancelled
                ))
            ));
            let sink = pipeline
                .into_sink()
                .into_any()
                .downcast::<TestSink>()
                .unwrap();
            assert!(sink.chunks.is_empty());
            assert!(!sink.finalized);
        }
    }

    #[test]
    fn cancellation_after_successful_push_prevents_downstream_work() {
        for variant in [false, true] {
            for intermediate in [false, true] {
                let control = QueryExecutionControl::new();
                let downstream_pushes = Arc::new(AtomicUsize::new(0));
                let downstream_finalizes = Arc::new(AtomicUsize::new(0));
                let mut pipeline =
                    Pipeline::simple(Box::new(TestSource::new(1, 1)), Box::new(TestSink::new()))
                        .with_operator(Box::new(CancellingPushOperator {
                            cancellation: control.cancellation_handle(),
                            cancel_on_push: true,
                            cancel_on_finalize: false,
                            fail: false,
                        }))
                        .with_execution_checkpoint(control.checkpoint());
                if intermediate {
                    pipeline = pipeline.with_operator(Box::new(ContinuationProbeOperator {
                        pushes: Arc::clone(&downstream_pushes),
                        finalizes: Arc::clone(&downstream_finalizes),
                        forward: true,
                        continue_after_push: true,
                        fail_on_finalize: false,
                    }));
                }
                assert!(matches!(
                    execute_test_pipeline(&mut pipeline, variant),
                    Err(OperatorError::QueryCancelled(
                        QueryCancellationError::Cancelled
                    ))
                ));
                assert_eq!(downstream_pushes.load(Ordering::Acquire), 0);
                assert_eq!(downstream_finalizes.load(Ordering::Acquire), 0);
                let sink = pipeline
                    .into_sink()
                    .into_any()
                    .downcast::<TestSink>()
                    .unwrap();
                // An indivisible terminal push may deliver before returning.
                assert_eq!(sink.chunks.len(), usize::from(!intermediate));
                assert!(!sink.finalized);
            }
        }
    }

    #[test]
    fn cancellation_after_successful_sink_consume_prevents_another_pull() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let polls = Arc::new(AtomicUsize::new(0));
            let source = ObservedSource {
                source: TestSource::new(2, 1),
                polls: Arc::clone(&polls),
                fail: false,
            };
            let sink = CancellingSink {
                cancellation: control.cancellation_handle(),
                cancel_on_consume: true,
                cancel_on_finalize: false,
                fail: false,
            };
            let mut pipeline = Pipeline::simple(Box::new(source), Box::new(sink))
                .with_execution_checkpoint(control.checkpoint());
            for _ in 0..2 {
                assert!(matches!(
                    execute_test_pipeline(&mut pipeline, variant),
                    Err(OperatorError::QueryCancelled(
                        QueryCancellationError::Cancelled
                    ))
                ));
                assert_eq!(polls.load(Ordering::Acquire), 1);
            }
        }
    }

    #[test]
    fn cancellation_after_successful_operator_finalize_prevents_downstream_finalization() {
        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let downstream_pushes = Arc::new(AtomicUsize::new(0));
            let downstream_finalizes = Arc::new(AtomicUsize::new(0));
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(0, 1)), Box::new(TestSink::new()))
                    .with_operator(Box::new(CancellingPushOperator {
                        cancellation: control.cancellation_handle(),
                        cancel_on_push: false,
                        cancel_on_finalize: true,
                        fail: false,
                    }))
                    .with_operator(Box::new(ContinuationProbeOperator {
                        pushes: Arc::clone(&downstream_pushes),
                        finalizes: Arc::clone(&downstream_finalizes),
                        forward: true,
                        continue_after_push: true,
                        fail_on_finalize: false,
                    }))
                    .with_execution_checkpoint(control.checkpoint());
            assert!(matches!(
                execute_test_pipeline(&mut pipeline, variant),
                Err(OperatorError::QueryCancelled(
                    QueryCancellationError::Cancelled
                ))
            ));
            assert_eq!(downstream_pushes.load(Ordering::Acquire), 0);
            assert_eq!(downstream_finalizes.load(Ordering::Acquire), 0);
            let sink = pipeline
                .into_sink()
                .into_any()
                .downcast::<TestSink>()
                .unwrap();
            assert!(sink.chunks.is_empty());
            assert!(!sink.finalized);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn checkpoint_deadline_precedes_source_poll_and_requires_fresh_execution() {
        use std::time::Duration;

        for variant in [false, true] {
            for (chunks, fail) in [(0, false), (1, false), (1, true)] {
                let polls = Arc::new(AtomicUsize::new(0));
                let control = QueryExecutionControl::with_timeout(Duration::ZERO).unwrap();
                let source = ObservedSource {
                    source: TestSource::new(chunks, 1),
                    polls: Arc::clone(&polls),
                    fail,
                };
                let mut pipeline = Pipeline::simple(Box::new(source), Box::new(TestSink::new()))
                    .with_execution_checkpoint(control.checkpoint());
                for _ in 0..2 {
                    assert!(matches!(
                        execute_test_pipeline(&mut pipeline, variant),
                        Err(OperatorError::QueryCancelled(
                            QueryCancellationError::DeadlineExceeded {
                                timeout: Some(Duration::ZERO),
                            }
                        ))
                    ));
                    assert_eq!(
                        polls.load(Ordering::Acquire),
                        0,
                        "expired execution never polls"
                    );
                }
                let fresh = QueryExecutionControl::new();
                pipeline.set_execution_checkpoint(fresh.checkpoint());
                let result = execute_test_pipeline(&mut pipeline, variant);
                if fail {
                    assert!(
                        matches!(result, Err(OperatorError::Execution(ref message)) if message == "source failed")
                    );
                    assert_eq!(polls.load(Ordering::Acquire), 1);
                } else {
                    result.unwrap();
                    assert_eq!(polls.load(Ordering::Acquire), chunks + 1);
                }
                let sink = pipeline
                    .into_sink()
                    .into_any()
                    .downcast::<TestSink>()
                    .unwrap();
                assert_eq!(
                    sink.chunks.iter().map(DataChunk::len).sum::<usize>(),
                    if fail { 0 } else { chunks }
                );
                assert_eq!(sink.finalized, !fail);
                assert_eq!(fresh.checkpoint().check(), Ok(()));
                assert_eq!(
                    control.checkpoint().check(),
                    Err(QueryCancellationError::DeadlineExceeded {
                        timeout: Some(Duration::ZERO),
                    })
                );
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn earliest_composed_deadline_retains_its_diagnostic_in_either_order() {
        use std::time::{Duration, Instant};

        let earlier = Instant::now().checked_sub(Duration::from_secs(2)).unwrap();
        let later = earlier.checked_add(Duration::from_secs(1)).unwrap();
        for variant in [false, true] {
            for earlier_first in [false, true] {
                for timeout in [None, Some(Duration::from_millis(12))] {
                    let control = QueryExecutionControl::new();
                    let bounds = if earlier_first {
                        [(earlier, timeout), (later, Some(Duration::from_millis(34)))]
                    } else {
                        [(later, Some(Duration::from_millis(34))), (earlier, timeout)]
                    };
                    let mut checkpoint = control.checkpoint();
                    for (deadline, diagnostic) in bounds {
                        checkpoint = checkpoint.with_additional_deadline(deadline, diagnostic);
                    }
                    let mut pipeline = Pipeline::simple(
                        Box::new(TestSource::new(0, 1)),
                        Box::new(TestSink::new()),
                    )
                    .with_execution_checkpoint(checkpoint);
                    assert!(matches!(
                        execute_test_pipeline(&mut pipeline, variant),
                        Err(OperatorError::QueryCancelled(QueryCancellationError::DeadlineExceeded {
                            timeout: actual,
                        })) if actual == timeout
                    ));
                    assert_eq!(
                        control.token().check(),
                        Err(QueryCancellationError::DeadlineExceeded { timeout })
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pipeline_cancellation_and_deadline_preserve_first_winner() {
        use std::time::Duration;

        for variant in [false, true] {
            for cancellation_first in [false, true] {
                let control = QueryExecutionControl::with_timeout(Duration::ZERO).unwrap();
                let checkpoint = control.checkpoint();
                let expected = if cancellation_first {
                    assert!(control.cancellation_handle().try_cancel());
                    QueryCancellationError::Cancelled
                } else {
                    let error = QueryCancellationError::DeadlineExceeded {
                        timeout: Some(Duration::ZERO),
                    };
                    assert_eq!(checkpoint.check(), Err(error));
                    assert!(!control.cancellation_handle().try_cancel());
                    error
                };
                let polls = Arc::new(AtomicUsize::new(0));
                let source = ObservedSource {
                    source: TestSource::new(1, 1),
                    polls: Arc::clone(&polls),
                    fail: false,
                };
                let mut pipeline = Pipeline::simple(Box::new(source), Box::new(TestSink::new()))
                    .with_execution_checkpoint(checkpoint);
                for _ in 0..2 {
                    assert!(matches!(
                        execute_test_pipeline(&mut pipeline, variant),
                        Err(OperatorError::QueryCancelled(actual)) if actual == expected
                    ));
                    assert_eq!(polls.load(Ordering::Acquire), 0);
                }
                assert_eq!(control.token().check(), Err(expected));
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn reinstalling_checkpoint_cannot_clear_recorded_terminal_reason() {
        use std::time::{Duration, Instant};

        for variant in [false, true] {
            let control = QueryExecutionControl::new();
            let timeout = Duration::from_millis(11);
            let checkpoint = control.checkpoint().with_additional_deadline(
                Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
                Some(timeout),
            );
            let expected = QueryCancellationError::DeadlineExceeded {
                timeout: Some(timeout),
            };
            let mut pipeline =
                Pipeline::simple(Box::new(TestSource::new(1, 1)), Box::new(TestSink::new()))
                    .with_execution_checkpoint(checkpoint);
            assert!(
                matches!(execute_test_pipeline(&mut pipeline, variant), Err(OperatorError::QueryCancelled(actual)) if actual == expected)
            );
            // The original deadline-free snapshot shares the terminal state.
            pipeline.set_execution_checkpoint(control.checkpoint());
            assert!(
                matches!(execute_test_pipeline(&mut pipeline, variant), Err(OperatorError::QueryCancelled(actual)) if actual == expected)
            );
            let sink = pipeline
                .into_sink()
                .into_any()
                .downcast::<TestSink>()
                .unwrap();
            assert!(sink.chunks.is_empty());
            assert!(!sink.finalized);
        }
    }

    #[test]
    fn test_chunk_size_hints() {
        assert_eq!(ChunkSizeHint::default(), ChunkSizeHint::Default);

        let source = Box::new(TestSource::new(1, 10));
        let sink = Box::new(TestSink::new());

        // Test with small hint operator
        struct SmallHintOp;
        impl PushOperator for SmallHintOp {
            fn push(
                &mut self,
                chunk: DataChunk,
                sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                sink.consume(chunk)
            }
            fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
                Ok(())
            }
            fn preferred_chunk_size(&self) -> ChunkSizeHint {
                ChunkSizeHint::Small
            }
            fn name(&self) -> &'static str {
                "SmallHint"
            }
        }

        let pipeline = Pipeline::simple(source, sink).with_operator(Box::new(SmallHintOp));

        let computed_size = pipeline.compute_chunk_size();
        assert!(computed_size <= SMALL_CHUNK_SIZE);
    }

    struct KeepFactorizedOp;

    impl PushOperator for KeepFactorizedOp {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn push_variant(
            &mut self,
            chunk: ChunkVariant,
            sink: &mut dyn Sink,
        ) -> Result<bool, OperatorError> {
            sink.consume_variant(chunk)
        }

        fn accepts_factorized(&self) -> bool {
            true
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "KeepFactorized"
        }
    }

    #[test]
    fn variant_collector_preserves_single_factorized() {
        use crate::execution::factorized_chunk::FactorizedChunk;
        use grafeo_common::types::LogicalType;

        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(10);
        let fact = FactorizedChunk::with_flat_level(vec![sources], vec!["src".into()]);
        let mut collector = VariantCollector::new();
        collector
            .consume_variant(ChunkVariant::factorized(fact))
            .unwrap();
        let out = collector.into_single_variant();
        assert!(out.is_factorized());
        assert_eq!(out.logical_row_count(), 1);
    }

    #[test]
    fn execute_variant_default_flattens() {
        let source = Box::new(TestSource::new(1, 4));
        let sink = Box::new(VariantCollector::new());
        let mut pipeline =
            Pipeline::simple(source, sink).with_operator(Box::new(PassThroughOperator));
        pipeline.execute_variant().unwrap();
    }

    #[test]
    fn keep_factorized_push_does_not_flatten() {
        use crate::execution::factorized_chunk::FactorizedChunk;
        use grafeo_common::types::LogicalType;

        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);
        sources.push_int64(2);
        let fact = FactorizedChunk::with_flat_level(vec![sources], vec!["src".into()]);
        let mut op = KeepFactorizedOp;
        let mut sink = VariantCollector::new();
        op.push_variant(ChunkVariant::factorized(fact), &mut sink)
            .unwrap();
        let out = sink.into_single_variant();
        assert!(out.is_factorized());
        assert_eq!(out.logical_row_count(), 2);
        assert!(op.accepts_factorized());
    }
}

#[cfg(test)]
mod accounted_transport_tests {
    use super::*;
    use crate::execution::accounted_chunk::accounted_chunk_for_test;
    use crate::execution::{
        AccountedDataChunk, CardinalityTrackingOperator, CardinalityTrackingSink, CollectorSink,
        CountingSink, LimitingSink, MaterializingSink, NullSink, QueryExecutionControl,
        QueryResourceContext, SharedAdaptiveContext, ValueVector,
    };
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::Value;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn one_row_chunk(value: i64) -> DataChunk {
        DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(value)])])
    }

    fn admitted_chunk(context: &QueryResourceContext, value: i64) -> AccountedDataChunk {
        let chunk = one_row_chunk(value);
        let granted_bytes = chunk.observed_column_capacity_bytes().unwrap().max(1);
        let grant = context.try_allocate(granted_bytes).unwrap();
        accounted_chunk_for_test(chunk, grant)
    }

    struct PollCountingSource {
        polls: Arc<AtomicUsize>,
    }

    impl Source for PollCountingSource {
        fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
            self.polls.fetch_add(1, Ordering::AcqRel);
            Ok(None)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "PollCountingSource"
        }
    }

    struct PlainOnlySink {
        plain_calls: Arc<AtomicUsize>,
    }

    impl Sink for PlainOnlySink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            self.plain_calls.fetch_add(1, Ordering::AcqRel);
            Ok(true)
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "PlainOnlySink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    struct PlainOnlyOperator {
        plain_calls: Arc<AtomicUsize>,
    }

    impl PushOperator for PlainOnlyOperator {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            self.plain_calls.fetch_add(1, Ordering::AcqRel);
            sink.consume(chunk)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "PlainOnlyOperator"
        }
    }

    struct SelfAttestingSink {
        accounted_calls: Arc<AtomicUsize>,
    }

    impl Sink for SelfAttestingSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            Ok(true)
        }

        fn consume_accounted(&mut self, _chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
            self.accounted_calls.fetch_add(1, Ordering::AcqRel);
            Ok(true)
        }

        fn admit_chunk_transport(&self, _input: ChunkTransport) -> Result<(), OperatorError> {
            Ok(())
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "SelfAttestingSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    struct SelfAttestingForwarder {
        accounted_calls: Arc<AtomicUsize>,
    }

    impl PushOperator for SelfAttestingForwarder {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn push_accounted(
            &mut self,
            chunk: AccountedDataChunk,
            sink: &mut dyn Sink,
        ) -> Result<bool, OperatorError> {
            self.accounted_calls.fetch_add(1, Ordering::AcqRel);
            sink.consume_accounted(chunk)
        }

        fn admit_chunk_transport(
            &self,
            input: ChunkTransport,
        ) -> Result<ChunkTransport, OperatorError> {
            Ok(input)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "SelfAttestingForwarder"
        }
    }

    struct SelfAttestingProducer;

    impl PushOperator for SelfAttestingProducer {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn admit_chunk_transport(
            &self,
            _input: ChunkTransport,
        ) -> Result<ChunkTransport, OperatorError> {
            Ok(ChunkTransport::MayBeAccounted)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "SelfAttestingProducer"
        }
    }

    struct DelegatingSelfAttestingSink {
        inner: NullSink,
        accounted_calls: Arc<AtomicUsize>,
    }

    impl Sink for DelegatingSelfAttestingSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            Ok(true)
        }

        fn consume_accounted(&mut self, _chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
            self.accounted_calls.fetch_add(1, Ordering::AcqRel);
            Ok(true)
        }

        fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
            self.inner.__accounted_sink_permit()
        }

        fn admit_chunk_transport(&self, _input: ChunkTransport) -> Result<(), OperatorError> {
            Ok(())
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "DelegatingSelfAttestingSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    struct DelegatingSelfAttestingForwarder {
        inner: AccountedForwarder,
        accounted_calls: Arc<AtomicUsize>,
    }

    impl PushOperator for DelegatingSelfAttestingForwarder {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            self.inner.push(chunk, sink)
        }

        fn push_accounted(
            &mut self,
            chunk: AccountedDataChunk,
            sink: &mut dyn Sink,
        ) -> Result<bool, OperatorError> {
            self.accounted_calls.fetch_add(1, Ordering::AcqRel);
            sink.consume_accounted(chunk)
        }

        fn __accounted_push_permit(&mut self) -> Option<AccountedPushPermit<'_>> {
            self.inner.__accounted_push_permit()
        }

        fn admit_chunk_transport(
            &self,
            input: ChunkTransport,
        ) -> Result<ChunkTransport, OperatorError> {
            Ok(input)
        }

        fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
            self.inner.finalize(sink)
        }

        fn name(&self) -> &'static str {
            "DelegatingSelfAttestingForwarder"
        }
    }

    struct QualifiedFinalizeProducer {
        chunks: VecDeque<AccountedDataChunk>,
    }

    impl QualifiedFinalizeProducer {
        fn new(chunks: impl IntoIterator<Item = AccountedDataChunk>) -> Self {
            Self {
                chunks: chunks.into_iter().collect(),
            }
        }
    }

    impl PushOperator for QualifiedFinalizeProducer {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
            while let Some(chunk) = self.chunks.pop_front() {
                let _ = accounted_sink_permit(sink)?.consume(chunk)?;
            }
            Ok(())
        }

        fn __accounted_push_permit(&mut self) -> Option<AccountedPushPermit<'_>> {
            Some(AccountedPushPermit::new(self))
        }

        fn admit_chunk_transport(
            &self,
            input: ChunkTransport,
        ) -> Result<ChunkTransport, OperatorError> {
            if input == ChunkTransport::MayBeAccounted {
                return Err(OperatorError::UnsupportedAccountedTransport {
                    consumer: self.name(),
                });
            }
            Ok(ChunkTransport::MayBeAccounted)
        }

        fn name(&self) -> &'static str {
            "QualifiedFinalizeProducer"
        }
    }

    impl qualified_accounted_transport::QualifiedPushOperator for QualifiedFinalizeProducer {
        fn push_accounted_qualified(
            &mut self,
            _chunk: AccountedDataChunk,
            _sink: &mut AccountedSinkPermit<'_>,
        ) -> Result<bool, OperatorError> {
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: self.name(),
            })
        }
    }

    struct AccountedForwarder;

    impl PushOperator for AccountedForwarder {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn push_accounted(
            &mut self,
            chunk: AccountedDataChunk,
            sink: &mut dyn Sink,
        ) -> Result<bool, OperatorError> {
            sink.consume_accounted(chunk)
        }

        fn __accounted_push_permit(&mut self) -> Option<AccountedPushPermit<'_>> {
            Some(AccountedPushPermit::new(self))
        }

        fn admit_chunk_transport(
            &self,
            input: ChunkTransport,
        ) -> Result<ChunkTransport, OperatorError> {
            Ok(input)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "AccountedForwarder"
        }
    }

    impl qualified_accounted_transport::QualifiedPushOperator for AccountedForwarder {
        fn push_accounted_qualified(
            &mut self,
            chunk: AccountedDataChunk,
            sink: &mut AccountedSinkPermit<'_>,
        ) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }
    }

    struct NarrowingAccountedForwarder;

    impl PushOperator for NarrowingAccountedForwarder {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn push_accounted(
            &mut self,
            chunk: AccountedDataChunk,
            sink: &mut dyn Sink,
        ) -> Result<bool, OperatorError> {
            sink.consume_accounted(chunk)
        }

        fn admit_chunk_transport(
            &self,
            _input: ChunkTransport,
        ) -> Result<ChunkTransport, OperatorError> {
            Ok(ChunkTransport::PlainOnly)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "NarrowingAccountedForwarder"
        }
    }

    struct RetainingAccountedSink {
        chunks: Vec<AccountedDataChunk>,
        stop_after_first: bool,
    }

    impl Sink for RetainingAccountedSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            panic!("qualified test must not strip the accounted envelope")
        }

        fn consume_accounted(&mut self, chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
            self.chunks.push(chunk);
            Ok(!self.stop_after_first)
        }

        fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
            Some(AccountedSinkPermit::new(self))
        }

        fn admit_chunk_transport(&self, _input: ChunkTransport) -> Result<(), OperatorError> {
            Ok(())
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "RetainingAccountedSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    impl qualified_accounted_transport::QualifiedSink for RetainingAccountedSink {
        fn consume_accounted_qualified(
            &mut self,
            chunk: AccountedDataChunk,
        ) -> Result<bool, OperatorError> {
            self.chunks.push(chunk);
            Ok(!self.stop_after_first)
        }
    }

    #[test]
    fn default_accounted_sink_rejects_without_plain_fallback_and_releases_grant() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let plain_calls = Arc::new(AtomicUsize::new(0));
        let chunk = admitted_chunk(&context, 1);
        assert!(manager.allocated() > 0);
        let mut sink = PlainOnlySink {
            plain_calls: Arc::clone(&plain_calls),
        };

        assert!(matches!(
            sink.consume_accounted(chunk),
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: "PlainOnlySink"
            })
        ));
        assert_eq!(plain_calls.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn default_accounted_operator_rejects_without_plain_fallback_and_releases_grant() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let plain_calls = Arc::new(AtomicUsize::new(0));
        let chunk = admitted_chunk(&context, 1);
        let mut operator = PlainOnlyOperator {
            plain_calls: Arc::clone(&plain_calls),
        };
        let mut sink = NullSink::new();

        assert!(matches!(
            operator.push_accounted(chunk, &mut sink),
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: "PlainOnlyOperator"
            })
        ));
        assert_eq!(plain_calls.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn incompatible_accounted_transport_is_rejected_before_source_poll() {
        for variant_execution in [false, true] {
            let polls = Arc::new(AtomicUsize::new(0));
            let plain_calls = Arc::new(AtomicUsize::new(0));
            let mut pipeline = Pipeline::simple(
                Box::new(PollCountingSource {
                    polls: Arc::clone(&polls),
                }),
                Box::new(PlainOnlySink {
                    plain_calls: Arc::clone(&plain_calls),
                }),
            )
            .with_operator(Box::new(QualifiedFinalizeProducer::new([])));

            let result = if variant_execution {
                pipeline.execute_variant()
            } else {
                pipeline.execute()
            };
            assert!(matches!(
                result,
                Err(OperatorError::UnsupportedAccountedTransport {
                    consumer: "PlainOnlySink"
                })
            ));
            assert_eq!(polls.load(Ordering::Acquire), 0);
            assert_eq!(plain_calls.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn public_sink_self_attestation_does_not_authorize_accounted_execution() {
        for variant_execution in [false, true] {
            let polls = Arc::new(AtomicUsize::new(0));
            let accounted_calls = Arc::new(AtomicUsize::new(0));
            let mut pipeline = Pipeline::simple(
                Box::new(PollCountingSource {
                    polls: Arc::clone(&polls),
                }),
                Box::new(SelfAttestingSink {
                    accounted_calls: Arc::clone(&accounted_calls),
                }),
            )
            .with_operator(Box::new(QualifiedFinalizeProducer::new([])));

            pipeline.validate_chunk_transport().unwrap();
            let result = if variant_execution {
                pipeline.execute_variant()
            } else {
                pipeline.execute()
            };

            assert!(matches!(
                result,
                Err(OperatorError::UnsupportedAccountedTransport {
                    consumer: "SelfAttestingSink"
                })
            ));
            assert_eq!(polls.load(Ordering::Acquire), 0);
            assert_eq!(accounted_calls.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn public_operator_self_attestation_does_not_authorize_accounted_execution() {
        for variant_execution in [false, true] {
            let polls = Arc::new(AtomicUsize::new(0));
            let accounted_calls = Arc::new(AtomicUsize::new(0));
            let mut pipeline = Pipeline::new(
                Box::new(PollCountingSource {
                    polls: Arc::clone(&polls),
                }),
                vec![
                    Box::new(QualifiedFinalizeProducer::new([])),
                    Box::new(SelfAttestingForwarder {
                        accounted_calls: Arc::clone(&accounted_calls),
                    }),
                ],
                Box::new(NullSink::new()),
            );

            pipeline.validate_chunk_transport().unwrap();
            let result = if variant_execution {
                pipeline.execute_variant()
            } else {
                pipeline.execute()
            };

            assert!(matches!(
                result,
                Err(OperatorError::UnsupportedAccountedTransport {
                    consumer: "SelfAttestingForwarder"
                })
            ));
            assert_eq!(polls.load(Ordering::Acquire), 0);
            assert_eq!(accounted_calls.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn public_plain_to_accounted_self_attestation_does_not_authorize_production() {
        for variant_execution in [false, true] {
            let polls = Arc::new(AtomicUsize::new(0));
            let mut pipeline = Pipeline::new(
                Box::new(PollCountingSource {
                    polls: Arc::clone(&polls),
                }),
                vec![Box::new(SelfAttestingProducer)],
                Box::new(NullSink::new()),
            );

            pipeline.validate_chunk_transport().unwrap();
            let result = if variant_execution {
                pipeline.execute_variant()
            } else {
                pipeline.execute()
            };

            assert!(matches!(
                result,
                Err(OperatorError::UnsupportedAccountedTransport {
                    consumer: "SelfAttestingProducer"
                })
            ));
            assert_eq!(polls.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn relayed_permit_invokes_bound_inner_sink_not_public_wrapper_override() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let accounted_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = Pipeline::simple(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            Box::new(DelegatingSelfAttestingSink {
                inner: NullSink::new(),
                accounted_calls: Arc::clone(&accounted_calls),
            }),
        )
        .with_operator(Box::new(QualifiedFinalizeProducer::new([admitted_chunk(
            &context, 1,
        )])));

        pipeline.execute().unwrap();

        assert_eq!(accounted_calls.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn relayed_operator_permit_invokes_bound_inner_not_public_wrapper_override() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let accounted_calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = Pipeline::new(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            vec![
                Box::new(QualifiedFinalizeProducer::new([admitted_chunk(
                    &context, 1,
                )])),
                Box::new(DelegatingSelfAttestingForwarder {
                    inner: AccountedForwarder,
                    accounted_calls: Arc::clone(&accounted_calls),
                }),
            ],
            Box::new(NullSink::new()),
        );

        pipeline.execute().unwrap();

        assert_eq!(accounted_calls.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn sealed_execution_traverses_counting_and_cardinality_wrappers() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let adaptive = SharedAdaptiveContext::new();
        let mut pipeline = Pipeline::new(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            vec![
                Box::new(QualifiedFinalizeProducer::new([admitted_chunk(
                    &context, 1,
                )])),
                Box::new(CardinalityTrackingOperator::new(
                    Box::new(AccountedForwarder),
                    "operator",
                    adaptive,
                )),
            ],
            Box::new(CardinalityTrackingSink::new(
                Box::new(CountingSink::new()),
                "sink",
                SharedAdaptiveContext::new(),
            )),
        );

        pipeline.execute().unwrap();

        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<CardinalityTrackingSink>()
            .unwrap();
        assert_eq!(sink.current_count(), 1);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn with_operator_cannot_bypass_monotone_execution_validation() {
        let polls = Arc::new(AtomicUsize::new(0));
        let plain_calls = Arc::new(AtomicUsize::new(0));
        let pipeline = Pipeline::simple(
            Box::new(PollCountingSource {
                polls: Arc::clone(&polls),
            }),
            Box::new(NullSink::new()),
        )
        .with_operator(Box::new(QualifiedFinalizeProducer::new([])));
        pipeline.validate_chunk_transport().unwrap();
        let mut pipeline = pipeline.with_operator(Box::new(PlainOnlyOperator {
            plain_calls: Arc::clone(&plain_calls),
        }));

        assert!(matches!(
            pipeline.execute(),
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: "PlainOnlyOperator"
            })
        ));
        assert_eq!(polls.load(Ordering::Acquire), 0);
        assert_eq!(plain_calls.load(Ordering::Acquire), 0);
    }

    #[test]
    fn admission_fold_cannot_be_narrowed_after_accounted_output_is_possible() {
        let plain_calls = Arc::new(AtomicUsize::new(0));
        let pipeline = Pipeline::new(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            vec![
                Box::new(QualifiedFinalizeProducer::new([])),
                Box::new(NarrowingAccountedForwarder),
            ],
            Box::new(PlainOnlySink { plain_calls }),
        );

        assert!(matches!(
            pipeline.validate_chunk_transport(),
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: "PlainOnlySink"
            })
        ));
    }

    #[test]
    fn streaming_relay_admission_folds_remaining_chain_monotonically() {
        let mut operators: Vec<Box<dyn PushOperator>> = vec![Box::new(NarrowingAccountedForwarder)];
        let plain_calls = Arc::new(AtomicUsize::new(0));
        let mut sink = PlainOnlySink { plain_calls };
        let mut progress = ChainProgress::Continue;
        let relay = StreamingPipelineSink {
            operators: &mut operators,
            sink: &mut sink,
            checkpoint: None,
            stopped: false,
            start_index: 0,
            progress: &mut progress,
        };

        assert!(matches!(
            relay.admit_chunk_transport(ChunkTransport::MayBeAccounted),
            Err(OperatorError::UnsupportedAccountedTransport {
                consumer: "PlainOnlySink"
            })
        ));
    }

    #[test]
    fn retaining_compatibility_collectors_remain_explicitly_unqualified() {
        let sinks: Vec<(Box<dyn Sink>, &'static str)> = vec![
            (Box::new(VariantCollector::new()), "VariantCollector"),
            (Box::new(ChunkCollector::new()), "ChunkCollector"),
            (Box::new(CollectorSink::new()), "CollectorSink"),
            (Box::new(MaterializingSink::new()), "MaterializingSink"),
            (Box::new(LimitingSink::new(1)), "LimitingSink"),
        ];

        for (sink, expected_consumer) in sinks {
            assert!(matches!(
                sink.admit_chunk_transport(ChunkTransport::MayBeAccounted),
                Err(OperatorError::UnsupportedAccountedTransport { consumer })
                    if consumer == expected_consumer
            ));
        }
    }

    #[test]
    fn recursive_accounted_finalization_retains_charge_until_sink_drop() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let chunk = admitted_chunk(&context, 1);
        let charged = chunk.granted_bytes();
        let producer = QualifiedFinalizeProducer::new([chunk]);
        drop(context);

        let mut pipeline = Pipeline::new(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            vec![Box::new(producer), Box::new(AccountedForwarder)],
            Box::new(RetainingAccountedSink {
                chunks: Vec::new(),
                stop_after_first: false,
            }),
        );
        pipeline.execute().unwrap();
        assert_eq!(manager.allocated(), charged);

        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<RetainingAccountedSink>()
            .unwrap();
        assert_eq!(sink.chunks.len(), 1);
        assert_eq!(sink.chunks[0].chunk().len(), 1);
        assert_eq!(manager.allocated(), charged);
        drop(sink);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn terminal_false_blocks_second_accounted_delivery_and_releases_its_grant() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let first = admitted_chunk(&context, 1);
        let retained_bytes = first.granted_bytes();
        let second = admitted_chunk(&context, 2);
        let producer = QualifiedFinalizeProducer::new([first, second]);
        drop(context);

        let mut pipeline = Pipeline::new(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            vec![Box::new(producer), Box::new(AccountedForwarder)],
            Box::new(RetainingAccountedSink {
                chunks: Vec::new(),
                stop_after_first: true,
            }),
        );
        pipeline.execute().unwrap();
        assert_eq!(manager.allocated(), retained_bytes);

        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<RetainingAccountedSink>()
            .unwrap();
        assert_eq!(sink.chunks.len(), 1);
        drop(sink);
        assert_eq!(manager.allocated(), 0);
    }

    struct CancellingErrorSink {
        cancellation: crate::execution::QueryCancellationHandle,
    }

    impl Sink for CancellingErrorSink {
        fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
            panic!("qualified test must not strip the accounted envelope")
        }

        fn consume_accounted(&mut self, _chunk: AccountedDataChunk) -> Result<bool, OperatorError> {
            self.cancellation.cancel();
            Err(OperatorError::Execution(
                "accounted sink failed".to_string(),
            ))
        }

        fn __accounted_sink_permit(&mut self) -> Option<AccountedSinkPermit<'_>> {
            Some(AccountedSinkPermit::new(self))
        }

        fn admit_chunk_transport(&self, _input: ChunkTransport) -> Result<(), OperatorError> {
            Ok(())
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "CancellingErrorSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    impl qualified_accounted_transport::QualifiedSink for CancellingErrorSink {
        fn consume_accounted_qualified(
            &mut self,
            _chunk: AccountedDataChunk,
        ) -> Result<bool, OperatorError> {
            self.cancellation.cancel();
            Err(OperatorError::Execution(
                "accounted sink failed".to_string(),
            ))
        }
    }

    #[test]
    fn accounted_sink_error_beats_racing_cancellation_and_releases_grant() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
        let chunk = admitted_chunk(&context, 1);
        let control = QueryExecutionControl::new();
        let mut pipeline = Pipeline::simple(
            Box::new(PollCountingSource {
                polls: Arc::new(AtomicUsize::new(0)),
            }),
            Box::new(CancellingErrorSink {
                cancellation: control.cancellation_handle(),
            }),
        )
        .with_operator(Box::new(QualifiedFinalizeProducer::new([chunk])))
        .with_execution_checkpoint(control.checkpoint());

        assert!(matches!(
            pipeline.execute(),
            Err(OperatorError::Execution(ref message)) if message == "accounted sink failed"
        ));
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn counting_null_and_cardinality_wrappers_preserve_accounted_transport() {
        let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
        let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();

        let mut counting = CountingSink::new();
        counting
            .consume_accounted(admitted_chunk(&context, 1))
            .unwrap();
        assert_eq!(counting.count(), 1);

        let mut null = NullSink::new();
        null.consume_accounted(admitted_chunk(&context, 2)).unwrap();

        let adaptive = SharedAdaptiveContext::new();
        let mut tracking_sink =
            CardinalityTrackingSink::new(Box::new(NullSink::new()), "sink", adaptive.clone());
        tracking_sink
            .consume_accounted(admitted_chunk(&context, 3))
            .unwrap();
        assert_eq!(tracking_sink.current_count(), 1);
        assert!(
            tracking_sink
                .admit_chunk_transport(ChunkTransport::MayBeAccounted)
                .is_ok()
        );

        let mut tracking_operator =
            CardinalityTrackingOperator::new(Box::new(AccountedForwarder), "operator", adaptive);
        tracking_operator
            .push_accounted(admitted_chunk(&context, 4), &mut null)
            .unwrap();
        assert_eq!(tracking_operator.current_count(), 1);
        assert_eq!(
            tracking_operator
                .admit_chunk_transport(ChunkTransport::MayBeAccounted)
                .unwrap(),
            ChunkTransport::MayBeAccounted
        );
        assert_eq!(manager.allocated(), 0);
    }

    #[derive(Clone, Copy, Debug)]
    enum ResultCallbackExit {
        Continue,
        Stop,
        Error,
        Unwind,
    }

    struct InputGrantProbe {
        manager: Arc<BufferManager>,
        expected_bytes: usize,
        dropped: Arc<AtomicUsize>,
    }

    impl Drop for InputGrantProbe {
        fn drop(&mut self) {
            // This destructor runs inside the callback's frame, including when
            // it unwinds. The input envelope must still own its whole grant.
            assert_eq!(self.manager.allocated(), self.expected_bytes);
            self.dropped.fetch_add(1, Ordering::AcqRel);
        }
    }

    struct GrantObservingResultConsumer {
        manager: Arc<BufferManager>,
        expected_bytes: usize,
        calls: Arc<AtomicUsize>,
        callback_drops: Arc<AtomicUsize>,
        exit: ResultCallbackExit,
    }

    impl ResultConsumer for GrantObservingResultConsumer {
        fn consume(&mut self, chunk: &DataChunk) -> Result<bool, OperatorError> {
            assert_eq!(self.manager.allocated(), self.expected_bytes);
            assert_eq!(chunk.row_count(), 1);
            assert_eq!(
                chunk.column(0).unwrap().get_value(0),
                Some(Value::Int64(73))
            );
            self.calls.fetch_add(1, Ordering::AcqRel);
            let _live_input = InputGrantProbe {
                manager: Arc::clone(&self.manager),
                expected_bytes: self.expected_bytes,
                dropped: Arc::clone(&self.callback_drops),
            };
            match self.exit {
                ResultCallbackExit::Continue => Ok(true),
                ResultCallbackExit::Stop => Ok(false),
                ResultCallbackExit::Error => {
                    Err(OperatorError::Execution("result callback failed".into()))
                }
                ResultCallbackExit::Unwind => panic!("result callback unwind witness"),
            }
        }

        fn consume_variant(&mut self, _chunk: &ChunkVariant) -> Result<bool, OperatorError> {
            panic!("accounted flat input must use the flat callback")
        }
    }

    #[test]
    fn result_consumer_sink_holds_input_grant_through_success_error_and_unwind() {
        for qualified in [false, true] {
            for exit in [
                ResultCallbackExit::Continue,
                ResultCallbackExit::Stop,
                ResultCallbackExit::Error,
                ResultCallbackExit::Unwind,
            ] {
                let manager = Arc::new(BufferManager::with_budget(1024 * 1024));
                let context = QueryResourceContext::new(Arc::clone(&manager)).unwrap();
                let chunk = admitted_chunk(&context, 73);
                let expected_bytes = manager.allocated();
                assert!(expected_bytes > 0);
                let calls = Arc::new(AtomicUsize::new(0));
                let callback_drops = Arc::new(AtomicUsize::new(0));
                let mut sink = ResultConsumerSink::new(GrantObservingResultConsumer {
                    manager: Arc::clone(&manager),
                    expected_bytes,
                    calls: Arc::clone(&calls),
                    callback_drops: Arc::clone(&callback_drops),
                    exit,
                });
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if qualified {
                        sink.__accounted_sink_permit().unwrap().consume(chunk)
                    } else {
                        sink.consume_accounted(chunk)
                    }
                }));
                match exit {
                    ResultCallbackExit::Continue => assert!(matches!(outcome, Ok(Ok(true)))),
                    ResultCallbackExit::Stop => assert!(matches!(outcome, Ok(Ok(false)))),
                    ResultCallbackExit::Error => assert!(
                        matches!(outcome, Ok(Err(OperatorError::Execution(ref message))) if message == "result callback failed")
                    ),
                    ResultCallbackExit::Unwind => assert!(outcome.is_err()),
                }
                assert_eq!(calls.load(Ordering::Acquire), 1);
                assert_eq!(callback_drops.load(Ordering::Acquire), 1);
                // The sink remains alive; input authority cannot be deferred
                // until sink destruction or terminal finalization.
                assert_eq!(manager.allocated(), 0, "{qualified:?} / {exit:?}");
                drop(sink);
                assert_eq!(manager.allocated(), 0);
            }
        }
    }

    struct FactorizedResultConsumer {
        calls: usize,
    }

    impl ResultConsumer for FactorizedResultConsumer {
        fn consume(&mut self, _chunk: &DataChunk) -> Result<bool, OperatorError> {
            panic!("factorized result input must not be flattened")
        }

        fn consume_variant(&mut self, chunk: &ChunkVariant) -> Result<bool, OperatorError> {
            let ChunkVariant::Factorized(chunk) = chunk else {
                panic!("variant callback must receive the original factorization")
            };
            assert_eq!(chunk.level_count(), 2);
            assert_eq!(chunk.logical_row_count(), 4);
            assert_eq!(chunk.physical_size(), 6);
            assert_eq!(chunk.level(1).unwrap().multiplicities(), &[3, 1]);
            self.calls += 1;
            Ok(true)
        }
    }

    #[test]
    fn result_consumer_sink_preserves_factorized_callback_and_multiplicity() {
        use crate::execution::factorized_chunk::FactorizedChunk;

        let mut chunk = FactorizedChunk::with_flat_level(
            vec![ValueVector::from_values(&[
                Value::Int64(10),
                Value::Int64(20),
            ])],
            vec!["parent".into()],
        );
        chunk.add_level(
            vec![ValueVector::from_values(&[
                Value::Int64(1),
                Value::Int64(2),
                Value::Int64(3),
                Value::Int64(4),
            ])],
            vec!["child".into()],
            &[0, 3, 4],
        );
        let mut sink = ResultConsumerSink::new(FactorizedResultConsumer { calls: 0 });
        assert!(
            sink.consume_variant(ChunkVariant::Factorized(chunk))
                .unwrap()
        );
        assert_eq!(sink.into_inner().calls, 1);
    }
}
