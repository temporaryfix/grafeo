//! Factorized expand operator for relationship traversal without row duplication.
//!
//! Unlike the regular [`ExpandOperator`](super::ExpandOperator) which duplicates input
//! rows for each neighbor, this operator keeps input data factorized and adds expansion
//! results as a new level. This provides massive memory and performance improvements
//! for multi-hop queries with high fan-out.
//!
//! # Example
//!
//! For a 2-hop query with 10 source nodes, each having 10 neighbors:
//!
//! - **Regular Expand**: 10 * 10 = 100 rows, each with duplicated source data
//! - **Factorized Expand**: 10 sources + 100 neighbors = 110 values, no duplication

use std::sync::Arc;

use super::{FactorizedOperator, Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::factorized_chunk::FactorizedChunk;
use crate::execution::vector::ValueVector;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashSet;

/// Result type for factorized operations.
pub type FactorizedResult = Result<Option<FactorizedChunk>, OperatorError>;

/// Records the complete structural edge predicate before adjacency
/// enumeration. Typed traversals use one guard per relationship type; an
/// untyped traversal conservatively guards the LPG dataset.
fn record_expand_predicate_read(
    store: &dyn GraphStoreSearch,
    edge_types: &[String],
    transaction_id: Option<TransactionId>,
) {
    let Some(tx) = transaction_id else {
        return;
    };
    if edge_types.is_empty() {
        store.record_lpg_dataset_read(tx);
    } else {
        for edge_type in edge_types {
            store.record_rel_type_predicate_read(tx, edge_type);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_dests_for_expand(
    store: &dyn GraphStoreSearch,
    source_id: NodeId,
    direction: Direction,
    edge_types: &[String],
    epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    use_versioned: bool,
    sip_allow: Option<&FxHashSet<NodeId>>,
    out: &mut Vec<NodeId>,
    edge_scratch: &mut Vec<(NodeId, EdgeId)>,
) {
    out.clear();
    // Omitting the stored Edge column must not collapse parallel relationships.
    // Backend neighbor-only APIs may return distinct destinations, so project
    // from the same visible, type/SIP-filtered edge rows as an edge-carrying hop.
    // The caller retains one pair buffer across all source rows, including on
    // current native reads; fill_neighbors_for_expand clears it before reuse.
    fill_neighbors_for_expand(
        store,
        source_id,
        direction,
        edge_types,
        epoch,
        transaction_id,
        use_versioned,
        sip_allow,
        edge_scratch,
    );
    out.extend(edge_scratch.iter().map(|(nid, _)| *nid));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_neighbors_for_expand(
    store: &dyn GraphStoreSearch,
    source_id: NodeId,
    direction: Direction,
    edge_types: &[String],
    epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    use_versioned: bool,
    sip_allow: Option<&FxHashSet<NodeId>>,
    out: &mut Vec<(NodeId, EdgeId)>,
) {
    out.clear();
    record_expand_predicate_read(store, edge_types, transaction_id);
    match (epoch, transaction_id) {
        (Some(ep), Some(transaction_id)) if use_versioned => {
            out.extend(store.edges_from_versioned(source_id, direction, ep, transaction_id));
        }
        // A real read epoch needs committed visibility even at the current
        // frontier: raw adjacency can contain another transaction's creates
        // or hide compact rows under its pending deletions.
        (Some(ep), _) if ep != EpochId::PENDING => {
            store.fill_edges_from_at_epoch(source_id, direction, ep, out);
        }
        (Some(_), _) | (None, _) => store.fill_edges_from(source_id, direction, out),
    }
    // Transport extracts may carry a source-owned edge before its destination
    // arrives in a sibling shard. Keep that structural row available for
    // export/merge, but never expose an unresolved endpoint as a query node.
    // The store discriminator keeps these point lookups off ordinary query
    // paths and remains sticky across receipt revocation for race safety.
    if store.may_have_unresolved_transport_edges() {
        out.retain(|(node_id, _)| match (epoch, transaction_id) {
            (Some(ep), Some(transaction_id)) if use_versioned => {
                store.is_node_visible_versioned(*node_id, ep, transaction_id)
            }
            (Some(ep), _) if ep != EpochId::PENDING => store.is_node_visible_at_epoch(*node_id, ep),
            (Some(_), _) | (None, _) => {
                store.is_node_visible_at_epoch(*node_id, store.current_epoch())
            }
        });
    }
    if let Some(allow) = sip_allow {
        out.retain(|(nid, _)| allow.contains(nid));
    }
    if edge_types.is_empty() || store.all_edges_have_types(edge_types) {
        return;
    }
    out.retain(|(_, edge_id)| {
        // Current `edge_type` is current-CSR only: a closed packed / sidecar
        // row (visible at a past epoch) returns None and would drop as-of
        // hops. Type is immutable, so as-of `get_edge_at_epoch` is the
        // source of truth when viewing a real epoch.
        let ty = match (epoch, transaction_id) {
            // Versioned adjacency can include an edge created by this owner
            // that has no committed epoch-only row yet. Keep the same view
            // when checking its type instead of filtering that edge back out.
            (Some(ep), Some(tx)) if use_versioned => store
                .get_edge_versioned(*edge_id, ep, tx)
                .map(|e| e.edge_type),
            (Some(ep), _) if ep != EpochId::PENDING => {
                store.get_edge_at_epoch(*edge_id, ep).map(|e| e.edge_type)
            }
            _ => store.edge_type(*edge_id),
        };
        ty.is_some_and(|actual| {
            edge_types
                .iter()
                .any(|t| actual.as_str().eq_ignore_ascii_case(t.as_str()))
        })
    });
}

fn hop_types_agree(a: &[String], b: &[String]) -> bool {
    a == b || a.is_empty() || b.is_empty()
}

fn collect_node_ids(
    source: &mut Box<dyn Operator>,
    column: usize,
    out: &mut Vec<NodeId>,
) -> Result<(), OperatorError> {
    while let Some(chunk) = source.next()? {
        let Some(col) = chunk.column(column) else {
            continue;
        };
        for i in 0..chunk.row_count() {
            if let Some(id) = col.get_node_id(i) {
                out.push(id);
            }
        }
    }
    Ok(())
}

fn count_two_hop_degree_product(store: &dyn GraphStoreSearch, _direction: Direction) -> u64 {
    // Directed 2-hop (a)→(b)→(c) and (a)←(b)←(c) are both Σ_b indeg(b)·outdeg(b).
    let mut n = 0u64;
    for id in store.node_ids() {
        n += store.in_degree(id) as u64 * store.out_degree(id) as u64;
    }
    n
}

fn count_two_hop_stream(
    store: &dyn GraphStoreSearch,
    seeds: &[NodeId],
    hop1: &ExpandStep,
    hop2: &ExpandStep,
    epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    use_versioned: bool,
) -> u64 {
    let sip1 = hop1.sip.as_ref().and_then(SipTarget::allowlist);
    let sip2 = hop2.sip.as_ref().and_then(SipTarget::allowlist);
    let current_degree =
        sip2.is_none() && !use_versioned && epoch.is_none_or(|ep| ep == EpochId::PENDING);
    let mut neighbors = Vec::new();
    let mut n = 0u64;
    for &src in seeds {
        fill_neighbors_for_expand(
            store,
            src,
            hop1.direction,
            &hop1.edge_types,
            epoch,
            transaction_id,
            use_versioned,
            sip1,
            &mut neighbors,
        );
        if current_degree {
            for (mid, _) in &neighbors {
                n += store.count_edges_from(*mid, hop2.direction, &hop2.edge_types) as u64;
            }
        } else {
            let hop1_neighbors = std::mem::take(&mut neighbors);
            for (mid, _) in hop1_neighbors {
                fill_neighbors_for_expand(
                    store,
                    mid,
                    hop2.direction,
                    &hop2.edge_types,
                    epoch,
                    transaction_id,
                    use_versioned,
                    sip2,
                    &mut neighbors,
                );
                n += neighbors.len() as u64;
            }
        }
    }
    n
}

/// An expand operator that produces factorized output.
///
/// Instead of duplicating input rows for each neighbor (Cartesian product),
/// this operator adds neighbors as a new factorization level. This avoids
/// exponential blowup in multi-hop queries.
///
/// # Memory Comparison
///
/// For a query `MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)` with:
/// - 100 source nodes
/// - Average 10 neighbors per hop
///
/// **Regular Expand (flat)**:
/// - After hop 1: 100 * 10 = 1,000 rows
/// - After hop 2: 1,000 * 10 = 10,000 rows
/// - Memory: ~10,000 * row_size
///
/// **Factorized Expand**:
/// - Level 0: 100 source nodes
/// - Level 1: 1,000 first-hop neighbors
/// - Level 2: 10,000 second-hop neighbors
/// - Memory: ~11,100 values (no duplication)
pub struct FactorizedExpandOperator {
    /// The store to traverse.
    store: Arc<dyn GraphStoreSearch>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    edge_types: Vec<String>,
    /// Transaction ID for MVCC visibility (None = use current epoch).
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Column names for the input (for tracking).
    input_column_names: Vec<String>,
    /// Optional SIP allow-list for produced neighbors.
    sip: Option<SipTarget>,
    /// Immediate (native default) or cooperative yield. Same expand kernel.
    scheduler: crate::execution::scheduler::Scheduler,
    /// When false, the new level is dest nodes only (no last-hop EdgeIds).
    need_edge: bool,
}

impl FactorizedExpandOperator {
    /// Creates a new factorized expand operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            exhausted: false,
            input_column_names: Vec::new(),
            sip: None,
            scheduler: crate::execution::scheduler::Scheduler::Immediate,
            need_edge: true,
        }
    }

    /// Last-hop dest nodes only — skip materializing [`EdgeId`]s.
    #[must_use]
    pub fn without_edge(mut self) -> Self {
        self.need_edge = false;
        self
    }

    /// Selects how the source loop yields. Default is
    /// [`crate::execution::scheduler::Scheduler::Immediate`] (native).
    #[must_use]
    pub fn with_scheduler(mut self, scheduler: crate::execution::scheduler::Scheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// Restricts emitted neighbors to `sip`'s allow-list.
    #[must_use]
    pub fn with_sip(mut self, sip: SipTarget) -> Self {
        self.sip = Some(sip);
        self
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this expand as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Sets the input column names for schema tracking.
    pub fn with_column_names(mut self, names: Vec<String>) -> Self {
        self.input_column_names = names;
        self
    }

    /// Processes an input chunk and produces a factorized chunk with expansion.
    fn process_chunk(&self, input: DataChunk) -> Result<FactorizedChunk, OperatorError> {
        let source_col = input.column(self.source_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
        })?;

        let row_count = input.row_count();

        let mut target_ids = ValueVector::with_type(LogicalType::Node);
        let mut offsets: Vec<u32> = Vec::with_capacity(row_count + 1);
        offsets.push(0);
        let mut row_err: Option<OperatorError> = None;
        if self.need_edge {
            let mut edge_ids = ValueVector::with_type(LogicalType::Edge);
            let mut neighbors = Vec::new();
            self.scheduler.for_each(
                row_count,
                |row_idx| {
                    if row_err.is_some() {
                        return;
                    }
                    let Some(source_id) = source_col.get_node_id(row_idx) else {
                        row_err = Some(OperatorError::Execution(
                            "Expected node ID in source column".into(),
                        ));
                        return;
                    };

                    fill_neighbors_for_expand(
                        self.store.as_ref(),
                        source_id,
                        self.direction,
                        &self.edge_types,
                        self.viewing_epoch,
                        self.transaction_id,
                        !self.read_only,
                        self.sip.as_ref().and_then(SipTarget::allowlist),
                        &mut neighbors,
                    );

                    for (target_id, edge_id) in &neighbors {
                        edge_ids.push_edge_id(*edge_id);
                        target_ids.push_node_id(*target_id);
                    }

                    // reason: factorized column lengths fit u32
                    #[allow(clippy::cast_possible_truncation)]
                    offsets.push(edge_ids.len() as u32);
                },
                crate::execution::scheduler::Scheduler::default_on_yield,
            );
            if let Some(err) = row_err {
                return Err(err);
            }
            let mut column_names: Vec<String> = if self.input_column_names.is_empty() {
                (0..input.column_count())
                    .map(|i| format!("col_{}", i))
                    .collect()
            } else {
                self.input_column_names.clone()
            };
            let mut chunk = FactorizedChunk::from_flat(&input, column_names.clone());
            if !edge_ids.is_empty() {
                column_names.push("_edge".to_string());
                column_names.push("_target".to_string());
                chunk.add_level(
                    vec![edge_ids, target_ids],
                    vec!["_edge".to_string(), "_target".to_string()],
                    &offsets,
                );
            }
            return Ok(chunk);
        }

        let mut dests = Vec::new();
        let mut edge_scratch = Vec::new();
        self.scheduler.for_each(
            row_count,
            |row_idx| {
                if row_err.is_some() {
                    return;
                }
                let Some(source_id) = source_col.get_node_id(row_idx) else {
                    row_err = Some(OperatorError::Execution(
                        "Expected node ID in source column".into(),
                    ));
                    return;
                };
                fill_dests_for_expand(
                    self.store.as_ref(),
                    source_id,
                    self.direction,
                    &self.edge_types,
                    self.viewing_epoch,
                    self.transaction_id,
                    !self.read_only,
                    self.sip.as_ref().and_then(SipTarget::allowlist),
                    &mut dests,
                    &mut edge_scratch,
                );
                for target_id in &dests {
                    target_ids.push_node_id(*target_id);
                }
                // reason: factorized column lengths fit u32
                #[allow(clippy::cast_possible_truncation)]
                offsets.push(target_ids.len() as u32);
            },
            crate::execution::scheduler::Scheduler::default_on_yield,
        );
        if let Some(err) = row_err {
            return Err(err);
        }
        let column_names: Vec<String> = if self.input_column_names.is_empty() {
            (0..input.column_count())
                .map(|i| format!("col_{}", i))
                .collect()
        } else {
            self.input_column_names.clone()
        };
        let mut chunk = FactorizedChunk::from_flat(&input, column_names);
        if !target_ids.is_empty() {
            chunk.add_level(vec![target_ids], vec!["_target".to_string()], &offsets);
        }
        Ok(chunk)
    }

    /// Gets the next factorized chunk.
    ///
    /// This is the main method for factorized execution. For compatibility
    /// with the regular `Operator` trait, use `next()` which flattens the result.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input operator or graph traversal fails.
    pub fn next_factorized(&mut self) -> FactorizedResult {
        if self.exhausted {
            return Ok(None);
        }

        match self.input.next() {
            Ok(Some(input)) => {
                let result = self.process_chunk(input)?;
                Ok(Some(result))
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

impl Operator for FactorizedExpandOperator {
    fn next(&mut self) -> OperatorResult {
        // For compatibility, flatten the factorized result
        match self.next_factorized() {
            Ok(Some(factorized)) => Ok(Some(factorized.flatten())),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "FactorizedExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn as_factorized_mut(&mut self) -> Option<&mut dyn FactorizedOperator> {
        Some(self)
    }
}

impl FactorizedOperator for FactorizedExpandOperator {
    fn next_factorized(&mut self) -> FactorizedResult {
        FactorizedExpandOperator::next_factorized(self)
    }
}

/// Builder for chaining multiple factorized expansions.
///
/// This is useful for multi-hop queries where you want to keep data
/// factorized across multiple expansion steps.
pub struct FactorizedExpandChain {
    /// The store to traverse.
    store: Arc<dyn GraphStoreSearch>,
    /// The source operator for the first expansion.
    source: Option<Box<dyn Operator>>,
    /// Accumulated factorized result.
    current_result: Option<FactorizedChunk>,
    /// Transaction context.
    transaction_id: Option<TransactionId>,
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
}

impl FactorizedExpandChain {
    /// Creates a new chain starting from a source operator.
    pub fn new(store: Arc<dyn GraphStoreSearch>, source: Box<dyn Operator>) -> Self {
        Self {
            store,
            source: Some(source),
            current_result: None,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
        }
    }

    /// Sets the transaction context.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this chain as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Adds an expansion step to the chain.
    ///
    /// Returns `self` for chaining.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the source operator or graph traversal fails.
    pub fn expand(
        self,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
    ) -> Result<Self, OperatorError> {
        self.expand_step(ExpandStep {
            source_column,
            direction,
            edge_types,
            sip: None,
            need_edge: true,
        })
    }

    /// Like [`Self::expand`] with a SIP allow-list on this hop.
    ///
    /// # Errors
    ///
    /// Returns [`OperatorError`] if the source operator fails or the expand cannot be planned.
    pub fn expand_step(mut self, step: ExpandStep) -> Result<Self, OperatorError> {
        let ExpandStep {
            source_column,
            direction,
            edge_types,
            sip,
            need_edge,
        } = step;
        // Get or create the initial factorized chunk
        if self.current_result.is_none() {
            if let Some(mut source) = self.source.take() {
                // Collect ALL batches from the source operator
                // This is necessary because the source may produce multiple batches
                let merged_input = Self::collect_all_batches(&mut *source)?;

                if let Some(input) = merged_input {
                    let mut expand = FactorizedExpandOperator::new(
                        Arc::clone(&self.store),
                        Box::new(SingleChunkOperator::new(input)),
                        source_column,
                        direction,
                        edge_types,
                    )
                    .with_read_only(self.read_only);
                    if !need_edge {
                        expand = expand.without_edge();
                    }

                    if let Some(s) = sip.clone() {
                        expand = expand.with_sip(s);
                    }

                    if let Some(epoch) = self.viewing_epoch {
                        expand = expand.with_transaction_context(epoch, self.transaction_id);
                    }

                    if let Some(result) = expand.next_factorized()? {
                        self.current_result = Some(result);
                    }
                }
            }
        } else {
            // Expand the deepest level of the factorized result
            // This adds a new level without flattening - the key to memory savings
            if let Some(mut factorized) = self.current_result.take() {
                let level_count_before = factorized.level_count();
                self.expand_deepest_level(
                    &mut factorized,
                    source_column,
                    direction,
                    edge_types,
                    sip.as_ref().and_then(SipTarget::allowlist),
                    need_edge,
                )?;
                if factorized.level_count() > level_count_before {
                    // New level was added: keep result
                    self.current_result = Some(factorized);
                }
                // Otherwise no edges were found at this level, so no valid
                // paths exist through the chain. Leave current_result as None
                // to signal an empty result set.
            }
        }

        Ok(self)
    }

    /// Collects all batches from an operator into a single merged chunk.
    fn collect_all_batches(source: &mut dyn Operator) -> Result<Option<DataChunk>, OperatorError> {
        let mut chunks: Vec<DataChunk> = Vec::new();

        while let Some(mut chunk) = source.next()? {
            // IMPORTANT: Flatten the chunk to materialize selection vectors.
            // FilterOperator returns chunks with selection vectors that logically filter rows,
            // but the underlying column data is unchanged. We need to physically copy only
            // the selected rows before using the data in factorized expansion.
            chunk.flatten();

            if chunk.row_count() > 0 {
                chunks.push(chunk);
            }
        }

        if chunks.is_empty() {
            return Ok(None);
        }

        if chunks.len() == 1 {
            return Ok(Some(chunks.remove(0)));
        }

        // Merge multiple chunks into one
        let first = &chunks[0];
        let total_rows: usize = chunks.iter().map(|c| c.row_count()).sum();
        let col_count = first.column_count();

        // Create merged vectors for each column
        let mut merged_cols: Vec<ValueVector> = (0..col_count)
            .map(|i| {
                let col_type = first
                    .column(i)
                    .map_or(&LogicalType::Any, |c| c.data_type())
                    .clone();
                ValueVector::with_type(col_type)
            })
            .collect();

        // Copy data from all chunks
        for chunk in &chunks {
            for col_idx in 0..col_count {
                if let Some(src_col) = chunk.column(col_idx) {
                    let dst_col = &mut merged_cols[col_idx];
                    for row_idx in 0..chunk.row_count() {
                        if let Some(value) = src_col.get_value(row_idx) {
                            dst_col.push_value(value);
                        }
                    }
                }
            }
        }

        let mut merged = DataChunk::new(merged_cols);
        merged.set_count(total_rows);

        Ok(Some(merged))
    }

    /// Expands the deepest level of a factorized chunk, adding a new level.
    ///
    /// This is the key method for multi-hop factorized execution. Instead of
    /// flattening and re-expanding, it directly processes the deepest level's
    /// target nodes and adds neighbors as a new level.
    fn expand_deepest_level(
        &self,
        chunk: &mut FactorizedChunk,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
        sip_allow: Option<&FxHashSet<NodeId>>,
        need_edge: bool,
    ) -> Result<(), OperatorError> {
        let epoch = self.viewing_epoch;
        let transaction_id = self.transaction_id;
        let use_versioned = !self.read_only;

        // Get the deepest level to find source nodes
        let deepest_level = chunk.level_count() - 1;
        let level = chunk
            .level(deepest_level)
            .ok_or_else(|| OperatorError::Execution("No levels in factorized chunk".into()))?;

        // Check if the source column exists in this level
        // If not, it means the previous expansion produced no edges (no level 1 was added)
        // In that case, there's nothing to expand further
        let Some(source_col) = level.column(source_column) else {
            // No source column means previous expand had no results
            // This is valid - just means no paths exist through this expansion
            return Ok(());
        };

        let mut target_ids = ValueVector::with_type(LogicalType::Node);
        let source_len = source_col.physical_len();
        let mut offsets: Vec<u32> = Vec::with_capacity(source_len + 1);
        offsets.push(0);

        if need_edge {
            let mut edge_ids = ValueVector::with_type(LogicalType::Edge);
            let mut neighbors = Vec::new();
            for idx in 0..source_len {
                let source_id = source_col.data().get_node_id(idx).ok_or_else(|| {
                    OperatorError::Execution("Expected node ID in source column".into())
                })?;

                fill_neighbors_for_expand(
                    self.store.as_ref(),
                    source_id,
                    direction,
                    &edge_types,
                    epoch,
                    transaction_id,
                    use_versioned,
                    sip_allow,
                    &mut neighbors,
                );

                for (target_id, edge_id) in &neighbors {
                    edge_ids.push_edge_id(*edge_id);
                    target_ids.push_node_id(*target_id);
                }

                // reason: factorized column lengths fit u32
                #[allow(clippy::cast_possible_truncation)]
                offsets.push(edge_ids.len() as u32);
            }
            if !edge_ids.is_empty() {
                chunk.add_level(
                    vec![edge_ids, target_ids],
                    vec!["_edge".to_string(), "_target".to_string()],
                    &offsets,
                );
            }
            return Ok(());
        }

        let mut dests = Vec::new();
        let mut edge_scratch = Vec::new();
        for idx in 0..source_len {
            let source_id = source_col.data().get_node_id(idx).ok_or_else(|| {
                OperatorError::Execution("Expected node ID in source column".into())
            })?;
            fill_dests_for_expand(
                self.store.as_ref(),
                source_id,
                direction,
                &edge_types,
                epoch,
                transaction_id,
                use_versioned,
                sip_allow,
                &mut dests,
                &mut edge_scratch,
            );
            for target_id in &dests {
                target_ids.push_node_id(*target_id);
            }
            // reason: factorized column lengths fit u32
            #[allow(clippy::cast_possible_truncation)]
            offsets.push(target_ids.len() as u32);
        }
        if !target_ids.is_empty() {
            chunk.add_level(vec![target_ids], vec!["_target".to_string()], &offsets);
        }

        Ok(())
    }

    /// Last-hop path count without storing neighbor vectors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn count_deepest(
        store: &dyn GraphStoreSearch,
        chunk: &FactorizedChunk,
        source_column: usize,
        direction: Direction,
        edge_types: &[String],
        epoch: Option<EpochId>,
        transaction_id: Option<TransactionId>,
        use_versioned: bool,
        sip_allow: Option<&FxHashSet<NodeId>>,
    ) -> Result<u64, OperatorError> {
        let Some(level) = chunk.level(chunk.level_count().saturating_sub(1)) else {
            return Ok(0);
        };
        let Some(source_col) = level.column(source_column) else {
            return Ok(0);
        };
        let current_count =
            sip_allow.is_none() && !use_versioned && epoch.is_none_or(|ep| ep == EpochId::PENDING);
        let mut neighbors = Vec::new();
        let mut count = 0u64;
        for idx in 0..source_col.physical_len() {
            let Some(source_id) = source_col.data().get_node_id(idx) else {
                return Err(OperatorError::Execution(
                    "Expected node ID in source column".into(),
                ));
            };
            if current_count {
                count += store.count_edges_from(source_id, direction, edge_types) as u64;
                continue;
            }
            fill_neighbors_for_expand(
                store,
                source_id,
                direction,
                edge_types,
                epoch,
                transaction_id,
                use_versioned,
                sip_allow,
                &mut neighbors,
            );
            count += neighbors.len() as u64;
        }
        Ok(count)
    }

    /// Finishes the chain and returns the factorized result.
    pub fn finish(self) -> Option<FactorizedChunk> {
        self.current_result
    }

    /// Finishes the chain and returns a flattened DataChunk.
    pub fn finish_flat(self) -> Option<DataChunk> {
        self.current_result.map(|c| c.flatten())
    }
}

/// Helper operator that returns a single chunk once.
struct SingleChunkOperator {
    chunk: Option<DataChunk>,
}

impl SingleChunkOperator {
    fn new(chunk: DataChunk) -> Self {
        Self { chunk: Some(chunk) }
    }
}

impl Operator for SingleChunkOperator {
    fn next(&mut self) -> OperatorResult {
        Ok(self.chunk.take())
    }

    fn reset(&mut self) {
        // Cannot reset - chunk is consumed
    }

    fn name(&self) -> &'static str {
        "SingleChunk"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Where a SIP allow-list came from (`design-proposals-factorization-wcoj` B).
#[derive(Clone, Debug)]
pub enum SipTarget {
    /// Restrict a scan-side seed set (planner: property-eq on the scan var).
    NodeScan {
        /// Allowed node ids.
        allow: FxHashSet<NodeId>,
    },
    /// Prune neighbors emitted by expand hop `hop` (0-based in the chain).
    ExpandHop {
        /// Index in the expand chain (0 = first hop).
        hop: usize,
        /// Allowed neighbor ids.
        allow: FxHashSet<NodeId>,
    },
    /// Join-built target set applied to hop `hop`.
    TargetSet {
        /// Index in the expand chain (0 = first hop).
        hop: usize,
        /// Allowed neighbor ids from the join build.
        allow: FxHashSet<NodeId>,
    },
}

impl SipTarget {
    /// Nodes this hint permits. `None` only if the variant is empty.
    #[must_use]
    pub fn allowlist(&self) -> Option<&FxHashSet<NodeId>> {
        match self {
            Self::NodeScan { allow }
            | Self::ExpandHop { allow, .. }
            | Self::TargetSet { allow, .. } => Some(allow),
        }
    }
}

/// Configuration for a single expand step in a lazy chain.
#[derive(Clone)]
pub struct ExpandStep {
    /// Source column index within the current level.
    pub source_column: usize,
    /// Direction of edge traversal.
    pub direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    pub edge_types: Vec<String>,
    /// Optional SIP allow-list for this hop's targets.
    pub sip: Option<SipTarget>,
    /// When false, this hop stores dest nodes only (anonymous last hop).
    pub need_edge: bool,
}

/// A lazy operator that executes a factorized expand chain when next() is called.
///
/// Unlike `FactorizedExpandChain` which executes immediately during construction,
/// this operator defers execution until query runtime. This is critical for
/// correctness when filters are applied above the expand chain.
///
/// # Factorized Aggregation Support
///
/// This operator supports returning factorized results via `next_factorized()`.
/// When the downstream operator can handle factorized data (e.g., factorized
/// aggregation), this avoids flattening and provides massive speedups.
pub struct LazyFactorizedChainOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreSearch>,
    /// The source operator (filter, scan, etc).
    source: Option<Box<dyn Operator>>,
    /// The expand steps to execute.
    steps: Vec<ExpandStep>,
    /// Transaction ID for MVCC visibility.
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Cached flat result after execution.
    result: Option<DataChunk>,
    /// Cached factorized result after execution.
    factorized_result: Option<FactorizedChunk>,
    /// Whether execution has completed.
    executed: bool,
}

impl LazyFactorizedChainOperator {
    /// Creates a new lazy factorized chain operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        source: Box<dyn Operator>,
        steps: Vec<ExpandStep>,
    ) -> Self {
        Self {
            store,
            source: Some(source),
            steps,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            result: None,
            factorized_result: None,
            executed: false,
        }
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this operator as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Applies a SIP hint to the matching hop (or replaces the scan seed).
    #[must_use]
    pub fn with_sip(mut self, sip: SipTarget) -> Self {
        match &sip {
            SipTarget::NodeScan { allow } => {
                let nodes: Vec<NodeId> = allow.iter().copied().collect();
                self.source = Some(Box::new(super::NodeListOperator::new(nodes, 2048)));
            }
            SipTarget::ExpandHop { hop, .. } | SipTarget::TargetSet { hop, .. } => {
                if let Some(step) = self.steps.get_mut(*hop) {
                    step.sip = Some(sip);
                }
            }
        }
        self
    }

    /// Path count for COUNT(*) / COUNT(last hop): last hop is not stored.
    pub(crate) fn count_paths(&mut self) -> Result<u64, OperatorError> {
        if self.executed {
            return Ok(self
                .factorized_result
                .as_ref()
                .map_or(0, |c| c.logical_row_count() as u64));
        }
        self.executed = true;
        let Some(last) = self.steps.last().cloned() else {
            let Some(mut source) = self.source.take() else {
                return Ok(0);
            };
            let mut n = 0u64;
            while let Some(chunk) = source.next()? {
                n += chunk.row_count() as u64;
            }
            return Ok(n);
        };
        let Some(mut source) = self.source.take() else {
            return Ok(0);
        };

        // Two same-direction hops, no SIP: COUNT is Σ_b indeg(b)·outdeg(b)
        // when the scan is every node and the type filter covers the store.
        // Fair-card hop2_all is this shape (2k×8 → 128k paths).
        if self.steps.len() == 2
            && last.sip.is_none()
            && self.steps[0].sip.is_none()
            && self.steps[0].direction == last.direction
            && matches!(last.direction, Direction::Outgoing | Direction::Incoming)
            && hop_types_agree(&self.steps[0].edge_types, &last.edge_types)
        {
            let mut seeds = Vec::new();
            collect_node_ids(&mut source, self.steps[0].source_column, &mut seeds)?;
            let types = if last.edge_types.is_empty() {
                &self.steps[0].edge_types
            } else {
                &last.edge_types
            };
            // Read-only does not mean that other transactions have no pending
            // writes. Raw degree products cannot answer a real read epoch.
            let current = self.transaction_id.is_none()
                && self.viewing_epoch.is_none_or(|ep| ep == EpochId::PENDING);
            if current
                && seeds.len() == self.store.node_count()
                && self.store.all_edges_have_types(types)
            {
                return Ok(count_two_hop_degree_product(
                    self.store.as_ref(),
                    last.direction,
                ));
            }
            return Ok(count_two_hop_stream(
                self.store.as_ref(),
                &seeds,
                &self.steps[0],
                &last,
                self.viewing_epoch,
                self.transaction_id,
                !self.read_only,
            ));
        }

        let prior: Vec<ExpandStep> = self.steps[..self.steps.len() - 1].to_vec();
        let mut chain = FactorizedExpandChain::new(Arc::clone(&self.store), source)
            .with_read_only(self.read_only);
        if let Some(epoch) = self.viewing_epoch {
            chain = chain.with_transaction_context(epoch, self.transaction_id);
        }
        if prior.is_empty() {
            chain = chain
                .expand_step(last)
                .map_err(|e| OperatorError::Execution(format!("Factorized expand failed: {e}")))?;
            return Ok(chain.finish().map_or(0, |c| c.logical_row_count() as u64));
        }
        for step in &prior {
            chain = chain
                .expand_step(step.clone())
                .map_err(|e| OperatorError::Execution(format!("Factorized expand failed: {e}")))?;
        }
        let Some(chunk) = chain.finish() else {
            return Ok(0);
        };
        FactorizedExpandChain::count_deepest(
            self.store.as_ref(),
            &chunk,
            last.source_column,
            last.direction,
            &last.edge_types,
            self.viewing_epoch,
            self.transaction_id,
            !self.read_only,
            last.sip.as_ref().and_then(SipTarget::allowlist),
        )
    }

    /// Executes the chain and returns the factorized result.
    ///
    /// This is the key method for factorized aggregation - it returns the
    /// factorized chunk without flattening, allowing O(n) aggregation instead
    /// of O(n²) or worse.
    fn execute_factorized(&mut self) -> Result<Option<FactorizedChunk>, OperatorError> {
        let Some(source) = self.source.take() else {
            return Ok(None);
        };

        // Build and execute the chain
        let mut chain = FactorizedExpandChain::new(Arc::clone(&self.store), source)
            .with_read_only(self.read_only);

        if let Some(epoch) = self.viewing_epoch {
            chain = chain.with_transaction_context(epoch, self.transaction_id);
        }

        // Execute each expand step
        for step in &self.steps {
            chain = chain.expand_step(step.clone()).map_err(|e| {
                OperatorError::Execution(format!("Factorized expand failed: {}", e))
            })?;
        }

        // Return the factorized result (not flattened)
        Ok(chain.finish())
    }

    /// Returns the factorized result without flattening.
    ///
    /// Use this when the next operator can handle factorized data (e.g.,
    /// factorized aggregation). This is the key to 10-100x speedups for
    /// aggregate queries on multi-hop traversals.
    ///
    /// # Returns
    ///
    /// The factorized chunk, or None if exhausted or no results.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying chain execution fails.
    pub fn next_factorized(&mut self) -> FactorizedResult {
        if self.executed {
            return Ok(self.factorized_result.take());
        }

        self.executed = true;
        self.factorized_result = self.execute_factorized()?;
        Ok(self.factorized_result.take())
    }

    /// Executes the chain and returns the flattened result.
    fn execute(&mut self) -> Result<Option<DataChunk>, OperatorError> {
        // Use the factorized execution and then flatten
        let factorized = self.execute_factorized()?;
        Ok(factorized.map(|c| c.flatten()))
    }
}

impl Operator for LazyFactorizedChainOperator {
    fn next(&mut self) -> OperatorResult {
        if self.executed {
            return Ok(self.result.take());
        }

        self.executed = true;
        self.result = self.execute()?;
        Ok(self.result.take())
    }

    fn reset(&mut self) {
        // Cannot reset - source has been consumed
        self.result = None;
        self.factorized_result = None;
        self.executed = true;
    }

    fn name(&self) -> &'static str {
        "LazyFactorizedChain"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn as_factorized_mut(&mut self) -> Option<&mut dyn FactorizedOperator> {
        Some(self)
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::utils::hash::FxHashSet;

    #[test]
    fn test_factorized_expand_basic() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        // Alix knows Gus and Vincent
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

        let mut expand = FactorizedExpandOperator::new(
            store.clone(),
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
        );

        // Get factorized result
        let result = expand.next_factorized().unwrap();
        assert!(result.is_some());

        let chunk = result.unwrap();

        // Should have 2 levels: sources and neighbors
        assert_eq!(chunk.level_count(), 2);

        // Level 0 has 3 sources (Alix, Gus, Vincent)
        assert_eq!(chunk.level(0).unwrap().column_count(), 1);

        // Level 1 has edges and targets
        // Only Alix has outgoing KNOWS edges (to Gus and Vincent)
        // So we should have 2 edges total
        assert_eq!(chunk.level(1).unwrap().column_count(), 2);
    }

    #[test]
    fn test_factorized_vs_flat_equivalence() {
        let store = Arc::new(LpgStore::new().unwrap());

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        // Run factorized expand
        let scan1 = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut factorized_expand =
            FactorizedExpandOperator::new(store.clone(), scan1, 0, Direction::Outgoing, vec![]);

        let factorized_result = factorized_expand.next_factorized().unwrap().unwrap();
        let flat_from_factorized = factorized_result.flatten();

        // Run regular expand (using the factorized operator's flat interface)
        let scan2 = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut regular_expand =
            FactorizedExpandOperator::new(store.clone(), scan2, 0, Direction::Outgoing, vec![]);

        let flat_result = regular_expand.next().unwrap().unwrap();

        // Both should have the same row count
        assert_eq!(
            flat_from_factorized.row_count(),
            flat_result.row_count(),
            "Factorized and flat should produce same row count"
        );
    }

    #[test]
    fn test_factorized_expand_no_edges() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes with no edges
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

        let mut expand =
            FactorizedExpandOperator::new(store.clone(), scan, 0, Direction::Outgoing, vec![]);

        let result = expand.next_factorized().unwrap();
        assert!(result.is_some());

        let chunk = result.unwrap();
        // Should only have the source level (no expansion level added when no edges)
        assert_eq!(chunk.level_count(), 1);
    }

    #[test]
    fn two_hop_line_from_label_scan_matches_flat_count() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let n = FactorizedExpandChain::new(store.clone(), scan)
            .expand(0, Direction::Outgoing, vec!["KNOWS".into()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["KNOWS".into()])
            .unwrap()
            .finish()
            .map_or(0, |chunk| chunk.flatten().row_count());
        assert_eq!(n, 1, "only a-b-c is a 2-hop path");
    }

    #[test]
    fn test_factorized_chain_two_hop() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create a 2-hop graph: a -> b1, b2 -> c1, c2, c3, c4
        let a = store.create_node(&["Person"]);
        let b1 = store.create_node(&["Person"]);
        let b2 = store.create_node(&["Person"]);
        let c1 = store.create_node(&["Person"]);
        let c2 = store.create_node(&["Person"]);
        let c3 = store.create_node(&["Person"]);
        let c4 = store.create_node(&["Person"]);

        // a knows b1 and b2
        store.create_edge(a, b1, "KNOWS");
        store.create_edge(a, b2, "KNOWS");

        // b1 knows c1 and c2
        store.create_edge(b1, c1, "KNOWS");
        store.create_edge(b1, c2, "KNOWS");

        // b2 knows c3 and c4
        store.create_edge(b2, c3, "KNOWS");
        store.create_edge(b2, c4, "KNOWS");

        // Create source chunk with just node 'a'
        let mut source_chunk = DataChunk::with_capacity(&[LogicalType::Node], 1);
        source_chunk.column_mut(0).unwrap().push_node_id(a);
        source_chunk.set_count(1);

        let source = Box::new(SingleChunkOperator::new(source_chunk));

        // Build 2-hop chain
        let chain = FactorizedExpandChain::new(store.clone(), source)
            .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["KNOWS".to_string()]) // column 1 is target from first expand
            .unwrap();

        let result = chain.finish().expect("Should have result");

        // Should have 3 levels: source (a), hop1 (b1,b2), hop2 (c1,c2,c3,c4)
        assert_eq!(result.level_count(), 3);

        // Physical size: 1 (source) + 2+2 (hop1 edges+targets) + 4+4 (hop2 edges+targets) = 13
        // vs flat which would be 4 rows * 5 columns = 20
        assert_eq!(result.physical_size(), 13);

        // Logical row count should be 4 (4 paths: a->b1->c1, a->b1->c2, a->b2->c3, a->b2->c4)
        assert_eq!(result.logical_row_count(), 4);

        // Flatten and verify
        let flat = result.flatten();
        assert_eq!(flat.row_count(), 4);
    }

    #[test]
    fn test_factorized_expand_multi_edge_type_filter() {
        let store = Arc::new(LpgStore::new().unwrap());

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["City"]);

        // Mixed edge types
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "LIVES_IN");
        store.create_edge(gus, vincent, "WORKS_AT");

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

        // Filter for KNOWS and LIVES_IN (case-insensitive)
        let mut expand = FactorizedExpandOperator::new(
            store.clone(),
            scan,
            0,
            Direction::Outgoing,
            vec!["knows".to_string(), "lives_in".to_string()],
        );

        let result = expand.next_factorized().unwrap().unwrap();
        let flat = result.flatten();

        // From Alix: KNOWS (to Gus) and LIVES_IN (to Vincent) = 2 rows
        // From Gus: WORKS_AT is filtered out = 0 rows
        assert_eq!(flat.row_count(), 2);
    }

    #[test]
    fn test_factorized_memory_savings() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create a star graph: center connected to 10 leaves
        let center = store.create_node(&["Center"]);
        let mut leaves = Vec::new();
        for _ in 0..10 {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center, leaf, "POINTS_TO");
            leaves.push(leaf);
        }

        // Scan just the center
        let mut source_chunk = DataChunk::with_capacity(&[LogicalType::Node], 1);
        source_chunk.column_mut(0).unwrap().push_node_id(center);
        source_chunk.set_count(1);

        let single = Box::new(SingleChunkOperator::new(source_chunk));

        let mut expand =
            FactorizedExpandOperator::new(store.clone(), single, 0, Direction::Outgoing, vec![]);

        let factorized = expand.next_factorized().unwrap().unwrap();

        // Physical size should be 1 (source) + 10 (edges) + 10 (targets) = 21 values
        // vs flat which would be 10 rows * 3 columns = 30 values
        assert_eq!(factorized.physical_size(), 21);

        // But logical row count should be 10
        assert_eq!(factorized.logical_row_count(), 10);

        // Flatten and verify correctness
        let flat = factorized.flatten();
        assert_eq!(flat.row_count(), 10);
    }

    #[test]
    fn test_factorized_expand_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let op = FactorizedExpandOperator::new(store.clone(), scan, 0, Direction::Outgoing, vec![]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<FactorizedExpandOperator>().is_ok());
    }

    #[test]
    fn test_lazy_factorized_chain_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let op = LazyFactorizedChainOperator::new(store.clone(), scan, vec![]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<LazyFactorizedChainOperator>().is_ok());
    }

    /// 2-hop factorized expand at a past epoch must not see a closed edge.
    #[cfg(feature = "compact-store")]
    #[test]
    fn factorized_two_hop_asof_excludes_closed_edge() {
        use crate::graph::compact::from_graph_store_preserving_ids;
        use crate::graph::compact::layered::LayeredStore;

        let empty = from_graph_store_preserving_ids(&LpgStore::new().unwrap()).unwrap();
        let layered = LayeredStore::new(empty, 16, 16).unwrap();
        let overlay = layered.overlay_store();
        overlay.set_epoch(EpochId::new(10));
        let a = overlay.create_node(&["Person"]);
        let b = overlay.create_node(&["Person"]);
        let c = overlay.create_node(&["Person"]);
        overlay.create_edge_versioned(a, b, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        let bc =
            overlay.create_edge_versioned(b, c, "KNOWS", EpochId::new(10), TransactionId::SYSTEM);
        overlay.set_epoch(EpochId::new(40));
        assert!(overlay.delete_edge(bc));
        overlay.set_epoch(EpochId::new(40));
        layered.merge_overlay_temporal().unwrap();

        let store: Arc<dyn GraphStoreSearch> = Arc::new(layered);
        let mut source = DataChunk::with_capacity(&[LogicalType::Node], 1);
        source.column_mut(0).unwrap().push_node_id(a);
        source.set_count(1);

        let asof = FactorizedExpandChain::new(
            Arc::clone(&store),
            Box::new(SingleChunkOperator::new(source)),
        )
        .with_transaction_context(EpochId::new(15), None)
        .with_read_only(true)
        .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
        .unwrap()
        .expand(1, Direction::Outgoing, vec!["KNOWS".to_string()])
        .unwrap()
        .finish()
        .expect("A->B->C visible at epoch 15");
        assert_eq!(
            asof.logical_row_count(),
            1,
            "as-of 15 must include closed-at-40 B->C"
        );

        let mut source_now = DataChunk::with_capacity(&[LogicalType::Node], 1);
        source_now.column_mut(0).unwrap().push_node_id(a);
        source_now.set_count(1);
        let now = FactorizedExpandChain::new(store, Box::new(SingleChunkOperator::new(source_now)))
            .with_transaction_context(EpochId::PENDING, None)
            .with_read_only(true)
            .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .finish();
        assert!(
            now.is_none() || now.as_ref().is_some_and(|c| c.logical_row_count() == 0),
            "current view must not include B->C deleted at 40"
        );
    }

    #[test]
    fn sip_expand_hop_prunes_neighbors() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let keep = store.create_node(&["V"]);
        let drop = store.create_node(&["V"]);
        store.create_edge(a, keep, "R");
        store.create_edge(a, drop, "R");

        let mut src = DataChunk::with_capacity(&[LogicalType::Node], 1);
        src.column_mut(0).unwrap().push_node_id(a);
        src.set_count(1);

        let mut allow = FxHashSet::default();
        allow.insert(keep);
        let pruned =
            FactorizedExpandChain::new(store.clone(), Box::new(SingleChunkOperator::new(src)))
                .expand_step(ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".to_string()],
                    sip: Some(SipTarget::ExpandHop { hop: 0, allow }),
                    need_edge: true,
                })
                .unwrap()
                .finish()
                .expect("keep neighbor");
        assert_eq!(pruned.logical_row_count(), 1);

        let mut src2 = DataChunk::with_capacity(&[LogicalType::Node], 1);
        src2.column_mut(0).unwrap().push_node_id(a);
        src2.set_count(1);
        let full = FactorizedExpandChain::new(store, Box::new(SingleChunkOperator::new(src2)))
            .expand(0, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .finish()
            .expect("both neighbors");
        assert_eq!(full.logical_row_count(), 2);
    }

    #[test]
    fn lazy_chain_with_sip_prunes() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let keep = store.create_node(&["V"]);
        let drop = store.create_node(&["V"]);
        store.create_edge(a, keep, "R");
        store.create_edge(a, drop, "R");
        let scan = Box::new(ScanOperator::with_label(store.clone(), "V"));
        let mut allow = FxHashSet::default();
        allow.insert(keep);
        let mut op = LazyFactorizedChainOperator::new(
            store,
            scan,
            vec![ExpandStep {
                source_column: 0,
                direction: Direction::Outgoing,
                edge_types: vec!["R".to_string()],
                sip: None,
                need_edge: true,
            }],
        )
        .with_sip(SipTarget::ExpandHop { hop: 0, allow });
        let chunk = op.next_factorized().unwrap().unwrap();
        assert_eq!(chunk.logical_row_count(), 1);
    }

    #[test]
    fn sip_target_set_is_join_hook() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c_keep = store.create_node(&["V"]);
        let c_drop = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c_keep, "R");
        store.create_edge(b, c_drop, "R");

        let mut src = DataChunk::with_capacity(&[LogicalType::Node], 1);
        src.column_mut(0).unwrap().push_node_id(a);
        src.set_count(1);
        let mut allow = FxHashSet::default();
        allow.insert(c_keep);
        let out = FactorizedExpandChain::new(store, Box::new(SingleChunkOperator::new(src)))
            .expand(0, Direction::Outgoing, vec!["R".to_string()])
            .unwrap()
            .expand_step(ExpandStep {
                source_column: 1,
                direction: Direction::Outgoing,
                edge_types: vec!["R".to_string()],
                sip: Some(SipTarget::TargetSet { hop: 1, allow }),
                need_edge: true,
            })
            .unwrap()
            .finish()
            .expect("join-bound c");
        assert_eq!(out.logical_row_count(), 1);
    }

    #[test]
    fn count_paths_two_hop_all_sources_matches_walk() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        // 3-cycle: each node is mid of one 2-hop. Count = 3.
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut op = LazyFactorizedChainOperator::new(
            store.clone(),
            scan,
            vec![
                ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    sip: None,
                    need_edge: true,
                },
                ExpandStep {
                    source_column: 1,
                    direction: Direction::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    sip: None,
                    need_edge: true,
                },
            ],
        )
        .with_read_only(true);
        assert_eq!(op.count_paths().unwrap(), 3);

        let mut product = 0u64;
        for id in store.node_ids() {
            product += store.in_degree(id) as u64 * store.out_degree(id) as u64;
        }
        assert_eq!(product, 3);
    }

    #[test]
    fn count_paths_two_hop_skips_identity_when_scan_is_subset() {
        let store = Arc::new(LpgStore::new().unwrap());
        let person = store.create_node(&["Person"]);
        let other = store.create_node(&["Other"]);
        let mid = store.create_node(&["Hub"]);
        store.create_edge(person, mid, "KNOWS");
        store.create_edge(other, mid, "KNOWS");
        store.create_edge(mid, person, "KNOWS");
        // Person starts: person→mid→person only (1). Identity over all nodes
        // would also count other→mid→person (2).
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut op = LazyFactorizedChainOperator::new(
            store,
            scan,
            vec![
                ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    sip: None,
                    need_edge: true,
                },
                ExpandStep {
                    source_column: 1,
                    direction: Direction::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    sip: None,
                    need_edge: true,
                },
            ],
        )
        .with_read_only(true);
        assert_eq!(op.count_paths().unwrap(), 1);
    }

    #[test]
    fn hop2_all_fair_card_shape_count() {
        const N: u32 = 2000;
        const DEG: u32 = 8;
        let store = Arc::new(LpgStore::new().unwrap());
        let mut ids = Vec::with_capacity(N as usize);
        for _ in 0..N {
            ids.push(store.create_node(&["Person"]));
        }
        let mut seen = std::collections::HashSet::new();
        let mut expect = 0u64;
        let dest = |src: u32, k: u32| {
            let mixed = src
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(k.wrapping_mul(0x85EB_CA6B));
            let mut out = mixed % N;
            if out == src {
                out = (out + 1) % N;
            }
            out
        };
        let mut adj = vec![Vec::new(); N as usize];
        for src in 0..N {
            for k in 0..DEG {
                let dst = dest(src, k);
                if seen.insert((src, dst)) {
                    store.create_edge(ids[src as usize], ids[dst as usize], "KNOWS");
                    adj[src as usize].push(dst);
                }
            }
        }
        for src in 0..N as usize {
            for &b in &adj[src] {
                expect += adj[b as usize].len() as u64;
            }
        }
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut op = LazyFactorizedChainOperator::new(
            store,
            scan,
            vec![
                ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    sip: None,
                    need_edge: true,
                },
                ExpandStep {
                    source_column: 1,
                    direction: Direction::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    sip: None,
                    need_edge: true,
                },
            ],
        )
        .with_read_only(true);
        assert_eq!(op.count_paths().unwrap(), expect);
        assert_eq!(expect, 128_000);
    }

    #[test]
    fn dest_only_last_hop_matches_with_edges() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");

        let scan = Box::new(ScanOperator::with_label(store.clone(), "V"));
        let with_edges = LazyFactorizedChainOperator::new(
            store.clone(),
            scan,
            vec![
                ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".to_string()],
                    sip: None,
                    need_edge: true,
                },
                ExpandStep {
                    source_column: 1,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".to_string()],
                    sip: None,
                    need_edge: true,
                },
            ],
        )
        .with_read_only(true)
        .next_factorized()
        .unwrap()
        .unwrap();

        let scan2 = Box::new(ScanOperator::with_label(store.clone(), "V"));
        let dest_only = LazyFactorizedChainOperator::new(
            store,
            scan2,
            vec![
                ExpandStep {
                    source_column: 0,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".to_string()],
                    sip: None,
                    need_edge: true,
                },
                ExpandStep {
                    source_column: 1,
                    direction: Direction::Outgoing,
                    edge_types: vec!["R".to_string()],
                    sip: None,
                    need_edge: false,
                },
            ],
        )
        .with_read_only(true)
        .next_factorized()
        .unwrap()
        .unwrap();

        assert_eq!(
            with_edges.logical_row_count(),
            dest_only.logical_row_count()
        );
        let last = dest_only.level_count() - 1;
        assert_eq!(
            dest_only.level(last).map(|l| l.column_count()),
            Some(1),
            "dest-only last hop stores nodes, not EdgeIds"
        );
        assert_eq!(with_edges.level(last).map(|l| l.column_count()), Some(2));
    }
    #[cfg(feature = "compact-store")]
    fn parallel_dest_fixture() -> (Arc<LpgStore>, NodeId, NodeId) {
        let store = Arc::new(LpgStore::new().unwrap());
        store.set_epoch(EpochId::new(10));
        let a = store.create_node(&["V"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        let excluded = store.create_node(&["V"]);
        for _ in 0..2 {
            store.create_edge(a, b, "R");
        }
        for _ in 0..3 {
            store.create_edge(b, c, "R");
        }
        store.create_edge(b, excluded, "OTHER");
        store.set_epoch(EpochId::new(40));
        store.create_edge(b, c, "R");
        (store, a, c)
    }

    #[cfg(feature = "compact-store")]
    fn assert_parallel_dest_multiplicity(
        store: Arc<dyn GraphStoreSearch>,
        source: NodeId,
        target: NodeId,
        epoch: EpochId,
        types: Vec<String>,
        expected: usize,
    ) {
        let mut projected_rows = Vec::new();
        for need_edge in [true, false] {
            let mut input = DataChunk::with_capacity(&[LogicalType::Node], 2);
            // Two identical input rows are two independent prefixes, and the
            // first hop has two parallel edges. Neither may be deduplicated.
            for _ in 0..2 {
                input.column_mut(0).unwrap().push_node_id(source);
            }
            input.set_count(2);
            let mut allow = FxHashSet::default();
            allow.insert(target);
            let factorized = LazyFactorizedChainOperator::new(
                store.clone(),
                Box::new(SingleChunkOperator::new(input)),
                vec![
                    ExpandStep {
                        source_column: 0,
                        direction: Direction::Outgoing,
                        edge_types: vec!["R".into()],
                        sip: None,
                        need_edge: true,
                    },
                    ExpandStep {
                        source_column: 1,
                        direction: Direction::Outgoing,
                        edge_types: types.clone(),
                        sip: Some(SipTarget::ExpandHop { hop: 1, allow }),
                        need_edge,
                    },
                ],
            )
            .with_transaction_context(epoch, None)
            .with_read_only(true)
            .next_factorized()
            .unwrap()
            .unwrap();
            assert_eq!(
                factorized.logical_row_count(),
                expected,
                "need_edge={need_edge}, epoch={epoch:?}: omitting an Edge column must preserve one row per edge"
            );
            assert_eq!(
                factorized.level(2).unwrap().column_count(),
                if need_edge { 2 } else { 1 }
            );
            let flat = factorized.flatten();
            assert_eq!(flat.row_count(), expected);
            let mut rows = Vec::new();
            let mut last_edges = FxHashSet::default();
            for row in flat.selected_indices() {
                rows.push((
                    flat.column(0).unwrap().get_node_id(row).unwrap(),
                    flat.column(1).unwrap().get_edge_id(row).unwrap(),
                    flat.column(2).unwrap().get_node_id(row).unwrap(),
                    flat.column(if need_edge { 4 } else { 3 })
                        .unwrap()
                        .get_node_id(row)
                        .unwrap(),
                ));
                if need_edge {
                    last_edges.insert(flat.column(3).unwrap().get_edge_id(row).unwrap());
                }
            }
            if need_edge {
                assert_eq!(last_edges.len(), expected / 4);
            }
            rows.sort_unstable();
            projected_rows.push(rows);
        }
        assert_eq!(projected_rows[0], projected_rows[1]);
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn dest_only_parallel_multiplicity_native_current_and_snapshot_controls() {
        let (store, a, c) = parallel_dest_fixture();
        assert_parallel_dest_multiplicity(store.clone(), a, c, EpochId::new(40), vec![], 16);
        assert_parallel_dest_multiplicity(store, a, c, EpochId::new(15), vec!["R".into()], 12);
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn dest_only_parallel_multiplicity_hot_layered_current() {
        use crate::graph::compact::{CompactStoreBuilder, layered::LayeredStore};
        let (store, a, c) = parallel_dest_fixture();
        let base = Arc::new(CompactStoreBuilder::new().build().unwrap());
        let layered: Arc<dyn GraphStoreSearch> =
            Arc::new(LayeredStore::with_overlay(base, store).unwrap());
        // A typed fallback already uses edge-bearing enumeration; it must stay
        // equivalent to the anonymous untyped fast path, including SIP filtering.
        assert_parallel_dest_multiplicity(
            layered.clone(),
            a,
            c,
            EpochId::new(40),
            vec!["R".into()],
            16,
        );
        assert_parallel_dest_multiplicity(layered, a, c, EpochId::new(40), vec![], 16);
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn dest_only_parallel_multiplicity_compact_snapshot() {
        use crate::graph::compact::layered::LayeredStore;
        let (store, a, c) = parallel_dest_fixture();
        let layered = LayeredStore::from_native_temporal(store).unwrap();
        let compact: Arc<dyn GraphStoreSearch> = layered.base_store_arc();
        assert_ne!(compact.current_epoch(), EpochId::new(15));
        assert_parallel_dest_multiplicity(compact, a, c, EpochId::new(15), vec!["R".into()], 12);
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn dest_only_parallel_multiplicity_hot_layered_snapshot() {
        use crate::graph::compact::{CompactStoreBuilder, layered::LayeredStore};
        let (store, a, c) = parallel_dest_fixture();
        let base = Arc::new(CompactStoreBuilder::new().build().unwrap());
        let layered: Arc<dyn GraphStoreSearch> =
            Arc::new(LayeredStore::with_overlay(base, store).unwrap());
        assert_parallel_dest_multiplicity(layered, a, c, EpochId::new(15), vec!["R".into()], 12);
    }
    #[cfg(feature = "compact-store")]
    #[test]
    fn dest_only_parallel_multiplicity_first_level_backends() {
        use crate::graph::compact::{CompactStoreBuilder, layered::LayeredStore};
        let (native, a, _) = parallel_dest_fixture();
        let b = native.edges_from(a, Direction::Outgoing).next().unwrap().0;
        let hot: Arc<dyn GraphStoreSearch> = Arc::new(
            LayeredStore::with_overlay(
                Arc::new(CompactStoreBuilder::new().build().unwrap()),
                native,
            )
            .unwrap(),
        );
        let (native, compact_a, _) = parallel_dest_fixture();
        let compact_b = native
            .edges_from(compact_a, Direction::Outgoing)
            .next()
            .unwrap()
            .0;
        let compact: Arc<dyn GraphStoreSearch> = LayeredStore::from_native_temporal(native)
            .unwrap()
            .base_store_arc();
        for (backend, store, source, target, epoch) in [
            ("hot current", hot.clone(), a, b, EpochId::new(40)),
            ("hot asof", hot, a, b, EpochId::new(15)),
            (
                "compact asof",
                compact,
                compact_a,
                compact_b,
                EpochId::new(15),
            ),
        ] {
            let mut projected_rows = Vec::new();
            for need_edge in [true, false] {
                let mut input = DataChunk::with_capacity(&[LogicalType::Node], 2);
                for _ in 0..2 {
                    input.column_mut(0).unwrap().push_node_id(source);
                }
                input.set_count(2);
                let mut expand = FactorizedExpandOperator::new(
                    store.clone(),
                    Box::new(SingleChunkOperator::new(input)),
                    0,
                    Direction::Outgoing,
                    vec![],
                )
                .with_transaction_context(epoch, None)
                .with_read_only(true);
                if !need_edge {
                    expand = expand.without_edge();
                }
                let factorized = expand.next_factorized().unwrap().unwrap();
                assert_eq!(
                    factorized.logical_row_count(),
                    4,
                    "{backend}, need_edge={need_edge}"
                );
                assert_eq!(
                    factorized.level(1).unwrap().column_count(),
                    if need_edge { 2 } else { 1 }
                );
                let flat = factorized.flatten();
                assert_eq!(flat.column_count(), if need_edge { 3 } else { 2 });
                assert_eq!(flat.row_count(), 4);
                let mut rows = Vec::new();
                let mut edges = FxHashSet::default();
                for row in flat.selected_indices() {
                    let actual_source = flat.column(0).unwrap().get_node_id(row).unwrap();
                    let actual_target = flat
                        .column(if need_edge { 2 } else { 1 })
                        .unwrap()
                        .get_node_id(row)
                        .unwrap();
                    assert_eq!((actual_source, actual_target), (source, target));
                    rows.push((actual_source, actual_target));
                    if need_edge {
                        edges.insert(flat.column(1).unwrap().get_edge_id(row).unwrap());
                    }
                }
                if need_edge {
                    assert_eq!(edges.len(), 2);
                }
                projected_rows.push(rows);
            }
            assert_eq!(projected_rows[0], projected_rows[1], "{backend}");
        }
    }
}
