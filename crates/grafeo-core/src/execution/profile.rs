//! Query profiling infrastructure.
//!
//! Provides [`ProfiledOperator`], a wrapper that collects runtime statistics
//! (row counts, timing, call counts) around any pull-based [`Operator`].
//! Used by the `PROFILE` statement to annotate each operator with actual
//! execution metrics.

use std::sync::Arc;

use parking_lot::Mutex;

use super::operators::{Operator, OperatorResult};

/// Query-wide resource history, shared by all profiled operators.
/// Counters saturate on overflow; they are not per-operator attribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryProfileStats {
    /// Resident bytes granted across the query at the last operator sample.
    /// Sampling occurs after `next()`, `reset()` or engine finalization; later drops
    /// can release grants before the profile is formatted.
    pub resident_granted_bytes: usize,
    /// High-water mark of committed query resident grants.
    pub resident_peak_bytes: usize,
    /// Cumulative bytes of successfully published spill files, including merge outputs.
    pub spilled_bytes: u64,
    /// Cumulative published sort-run files, including intermediate runs.
    pub spill_runs: u64,
    /// Cumulative published native-partition files (not RDF state fragments).
    pub spill_partitions: u64,
    /// Cumulative active sort-merge work; unavailable on wasm32.
    pub merge_time_ns: Option<u64>,
    /// Query-local physical reservation and cleanup history, when authenticated.
    #[cfg(feature = "spill")]
    pub spill_physical: Option<super::spill::SpillPhysicalStats>,
    /// Last successful root-wide bounded cleanup pass, not current query usage.
    #[cfg(feature = "spill")]
    pub spill_recovery: Option<super::spill::SpillScavengeReport>,
}

/// Runtime statistics for a single operator in a profiled query.
#[derive(Debug, Clone, Default)]
pub struct ProfileStats {
    /// Total rows produced as output.
    pub rows_out: u64,
    /// Wall-clock time spent in this operator (nanoseconds), including children.
    pub time_ns: u64,
    /// Number of times `next()` was called on this operator.
    pub calls: u64,
    /// Latest query-wide snapshot; repeated nodes must not be summed.
    pub query_resources: Option<QueryProfileStats>,
}

/// Shared handle to profile stats, written by `ProfiledOperator` during
/// execution and read afterwards for formatting.
pub type SharedProfileStats = Arc<Mutex<ProfileStats>>;

/// Wraps a pull-based [`Operator`] to collect runtime statistics.
///
/// Each call to [`next()`](Operator::next) is timed and the output rows
/// are counted. Statistics are written into a [`SharedProfileStats`] handle
/// so they can be collected after execution completes.
pub struct ProfiledOperator {
    inner: Box<dyn Operator>,
    stats: SharedProfileStats,
    resources: Option<super::QueryResourceContext>,
}

impl ProfiledOperator {
    /// Creates a new profiled wrapper around the given operator.
    pub fn new(inner: Box<dyn Operator>, stats: SharedProfileStats) -> Self {
        Self {
            inner,
            stats,
            resources: None,
        }
    }
}

impl Operator for ProfiledOperator {
    fn next(&mut self) -> OperatorResult {
        {
            let mut s = self.stats.lock();
            s.calls += 1;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();

        let result = self.inner.next();

        #[cfg(not(target_arch = "wasm32"))]
        {
            // reason: per-call elapsed nanos fits u64 for any practical duration
            #[allow(clippy::cast_possible_truncation)]
            let elapsed = start.elapsed().as_nanos() as u64;
            self.stats.lock().time_ns += elapsed;
        }

        if let Ok(Some(ref chunk)) = result {
            self.stats.lock().rows_out += chunk.row_count() as u64;
        }

        if let Some(resources) = &self.resources {
            self.stats.lock().query_resources = Some(resources.profile_stats());
        }
        result
    }

    fn reset(&mut self) {
        self.inner.reset();
        if let Some(resources) = &self.resources {
            // Capture the post-reset live grant count while retaining the
            // cumulative spill and high-water counters for PROFILE output.
            self.stats.lock().query_resources = Some(resources.profile_stats());
        }
        self.resources = None;
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn install_resource_context(
        &mut self,
        resources: &super::memory::QueryResourceContext,
    ) -> Result<(), super::memory::QueryResourceContextError> {
        self.inner.install_resource_context(resources)?;
        #[cfg(feature = "spill")]
        resources.enable_spill_profile_merge();
        self.resources = Some(resources.clone());
        Ok(())
    }
}

// ProfiledOperator is Send + Sync because:
// - inner: Box<dyn Operator> is Send + Sync (trait bound)
// - stats: Arc<parking_lot::Mutex<ProfileStats>> is Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    // Called at compile time to verify the bounds hold.
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<ProfiledOperator>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunk;
    use crate::execution::vector::ValueVector;
    use crate::execution::{QueryExecutionId, QueryResourceContext, QueryResourceContextError};
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::types::LogicalType;

    #[test]
    fn query_peak_includes_released_growth_and_survives_operator_reset() {
        let memory = BufferManager::with_budget(4096);
        let resources = QueryResourceContext::new(memory.clone()).unwrap();
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(MockOperator::new(0, 0)), stats.clone());
        profiled.install_resource_context(&resources).unwrap();
        let mut grant = resources.try_allocate(128).unwrap();
        grant.try_resize(1024).unwrap();
        grant.try_resize(16).unwrap();
        drop(grant);
        assert!(profiled.next().unwrap().is_none());
        profiled.reset();
        let snapshot = stats.lock().query_resources.unwrap();
        assert_eq!(snapshot.resident_granted_bytes, 0);
        assert_eq!(snapshot.resident_peak_bytes, 1024);
        assert_eq!(memory.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    #[test]
    fn query_profile_preserves_published_runs_after_file_cleanup() {
        use crate::execution::spill::SpillFileRole;
        let directory = tempfile::tempdir().unwrap();
        let manager_fixture = crate::execution::spill::BorrowedSpillFixture::new(directory.path());
        let memory = BufferManager::with_budget(1 << 20);
        let (resources, manager) = manager_fixture
            .build_operator_resources(
                memory,
                crate::execution::QueryExecutionControl::new().token(),
            )
            .unwrap();
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(MockOperator::new(0, 0)), stats.clone());
        profiled.install_resource_context(&resources).unwrap();
        let mut file = manager.create_file(SpillFileRole::SortRun).unwrap();
        file.write_sort_run_start(1, 0).unwrap();
        file.finish_write().unwrap();
        let published = manager.spilled_bytes();
        assert!(published > 0);
        file.close_and_delete().unwrap();
        assert_eq!(manager.spilled_bytes(), 0);
        assert!(profiled.next().unwrap().is_none());
        profiled.reset();
        let snapshot = stats.lock().query_resources.unwrap();
        assert_eq!(snapshot.spilled_bytes, published);
        assert_eq!(snapshot.spill_runs, 1);
        assert_eq!(snapshot.spill_partitions, 0);
    }

    /// A mock operator that yields a fixed number of chunks, each with `rows_per_chunk` rows.
    struct MockOperator {
        chunks_remaining: usize,
        rows_per_chunk: usize,
    }

    impl MockOperator {
        fn new(chunks: usize, rows_per_chunk: usize) -> Self {
            Self {
                chunks_remaining: chunks,
                rows_per_chunk,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.chunks_remaining == 0 {
                return Ok(None);
            }
            self.chunks_remaining -= 1;
            let mut col = ValueVector::with_capacity(LogicalType::Int64, self.rows_per_chunk);
            // reason: test chunk rows are small, fit i64
            #[allow(clippy::cast_possible_wrap)]
            for i in 0..self.rows_per_chunk {
                col.push(grafeo_common::types::Value::Int64(i as i64));
            }
            let chunk = DataChunk::new(vec![col]);
            Ok(Some(chunk))
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "MockOperator"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    struct ResourceRecordingOperator {
        installed_query_ids: Arc<Mutex<Vec<QueryExecutionId>>>,
    }

    impl Operator for ResourceRecordingOperator {
        fn next(&mut self) -> OperatorResult {
            Ok(None)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "ProfileResourceRecorder"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }

        fn install_resource_context(
            &mut self,
            resources: &QueryResourceContext,
        ) -> Result<(), QueryResourceContextError> {
            self.installed_query_ids.lock().push(resources.query_id());
            Ok(())
        }
    }

    #[test]
    fn profile_stats_default_is_zero() {
        let stats = ProfileStats::default();
        assert_eq!(stats.rows_out, 0);
        assert_eq!(stats.time_ns, 0);
        assert_eq!(stats.calls, 0);
    }

    #[test]
    fn profiled_operator_counts_rows_and_calls() {
        let mock = MockOperator::new(3, 10);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));

        // Drain operator (3 chunks + 1 None = 4 calls)
        while profiled.next().unwrap().is_some() {}

        let s = stats.lock();
        assert_eq!(s.rows_out, 30); // 3 chunks x 10 rows
        assert_eq!(s.calls, 4); // 3 data + 1 None
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn profiled_operator_measures_time() {
        let mock = MockOperator::new(1, 5);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));

        let _ = profiled.next();
        assert!(stats.lock().time_ns > 0);
    }

    #[test]
    fn profiled_operator_delegates_name() {
        let mock = MockOperator::new(0, 0);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));
        assert_eq!(profiled.name(), "MockOperator");
    }

    #[test]
    fn profiled_operator_forwards_exact_resource_context_to_child() {
        let installed_query_ids = Arc::new(Mutex::new(Vec::new()));
        let child = ResourceRecordingOperator {
            installed_query_ids: Arc::clone(&installed_query_ids),
        };
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(child), stats);
        let resources = QueryResourceContext::new(BufferManager::with_budget(1024 * 1024))
            .expect("create query resources");

        profiled
            .install_resource_context(&resources)
            .expect("install profiled resources");

        assert_eq!(
            installed_query_ids.lock().as_slice(),
            &[resources.query_id()]
        );
    }
}
