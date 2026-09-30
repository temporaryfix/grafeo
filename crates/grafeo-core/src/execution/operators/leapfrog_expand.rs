//! Adjacency-native leapfrog expand for cyclic LPG patterns.
//!
//! Intersects two sorted neighbor runs with galloping seek instead of a
//! nested-loop third hop + hash probe. `CsrAdjacency` / packed targets are
//! table-local offsets and are **not** dest-sorted after original-id remap,
//! so this operator sorts each run once (cached per source).

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashMap;

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;

/// First index in `sorted[from..]` whose value is `>= key`.
#[must_use]
pub fn gallop_seek<T: Ord>(sorted: &[T], from: usize, key: &T) -> usize {
    if from >= sorted.len() {
        return sorted.len();
    }
    if &sorted[from] >= key {
        return from;
    }
    let mut step = 1usize;
    let mut idx = from;
    while let Some(probe) = idx.checked_add(step)
        && probe < sorted.len()
        && &sorted[probe] < key
    {
        idx = probe;
        step = step.saturating_mul(2);
    }
    let hi = idx.saturating_add(step).saturating_add(1).min(sorted.len());
    idx + sorted[idx..hi].partition_point(|x| x < key)
}

/// Number of shared values in two sorted unique slices.
#[must_use]
pub fn intersect_count<T: Ord>(left: &[T], right: &[T]) -> usize {
    let mut count = 0usize;
    let mut left_i = 0usize;
    let mut right_i = 0usize;
    while left_i < left.len() && right_i < right.len() {
        match left[left_i].cmp(&right[right_i]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                left_i += 1;
                right_i += 1;
            }
            std::cmp::Ordering::Less => left_i = gallop_seek(left, left_i + 1, &right[right_i]),
            std::cmp::Ordering::Greater => {
                right_i = gallop_seek(right, right_i + 1, &left[left_i]);
            }
        }
    }
    count
}

/// Appends the intersection of two sorted unique slices to `out`.
pub fn intersect_sorted<T: Copy + Ord>(a: &[T], b: &[T], out: &mut Vec<T>) {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i = gallop_seek(a, i + 1, &b[j]),
            std::cmp::Ordering::Greater => j = gallop_seek(b, j + 1, &a[i]),
        }
    }
}

/// Parallel neighbor run: `nodes[i]` reached via `edges[i]`, sorted by node.
#[derive(Clone, Default)]
struct NeighborRun {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
}

/// A resumable Cartesian product over equal-destination edge runs.
struct IntersectionRows {
    left: Arc<NeighborRun>,
    right: Arc<NeighborRun>,
    i: usize,
    j: usize,
    left_end: usize,
    right_end: usize,
    pair_left: usize,
    pair_right: usize,
}

impl IntersectionRows {
    fn new(left: Arc<NeighborRun>, right: Arc<NeighborRun>) -> Self {
        Self {
            left,
            right,
            i: 0,
            j: 0,
            left_end: 0,
            right_end: 0,
            pair_left: 0,
            pair_right: 0,
        }
    }

    fn next(&mut self) -> Option<(NodeId, EdgeId, EdgeId)> {
        loop {
            if self.pair_left < self.left_end {
                let result = (
                    self.left.nodes[self.pair_left],
                    self.left.edges[self.pair_left],
                    self.right.edges[self.pair_right],
                );
                self.pair_right += 1;
                if self.pair_right == self.right_end {
                    self.pair_right = self.j;
                    self.pair_left += 1;
                }
                if self.pair_left == self.left_end {
                    self.i = self.left_end;
                    self.j = self.right_end;
                }
                return Some(result);
            }
            if self.i == self.left.nodes.len() || self.j == self.right.nodes.len() {
                return None;
            }
            match self.left.nodes[self.i].cmp(&self.right.nodes[self.j]) {
                std::cmp::Ordering::Equal => {
                    let target = self.left.nodes[self.i];
                    self.left_end =
                        self.i + self.left.nodes[self.i..].partition_point(|n| *n == target);
                    self.right_end =
                        self.j + self.right.nodes[self.j..].partition_point(|n| *n == target);
                    self.pair_left = self.i;
                    self.pair_right = self.j;
                }
                std::cmp::Ordering::Less => {
                    self.i = gallop_seek(&self.left.nodes, self.i + 1, &self.right.nodes[self.j]);
                }
                std::cmp::Ordering::Greater => {
                    self.j = gallop_seek(&self.right.nodes, self.j + 1, &self.left.nodes[self.i]);
                }
            }
        }
    }
}

fn triangle_node_visible(
    store: &dyn GraphStoreSearch,
    node: NodeId,
    epoch: EpochId,
    tx: Option<TransactionId>,
) -> bool {
    match tx {
        Some(tx) => store.is_node_visible_versioned(node, epoch, tx),
        None => store.is_node_visible_at_epoch(node, epoch),
    }
}

/// Apply one visibility context to adjacency, endpoints, edge types and SSI reads.
fn triangle_neighbors(
    store: &dyn GraphStoreSearch,
    source: NodeId,
    direction: Direction,
    types: &[String],
    epoch: Option<EpochId>,
    tx: Option<TransactionId>,
) -> Vec<(NodeId, EdgeId)> {
    // PENDING is the explicit current view. A compact store's numeric
    // current_epoch can be synthetic, so never substitute it for this view.
    let epoch = match tx {
        Some(_) => epoch
            .filter(|ep| *ep != EpochId::PENDING)
            .unwrap_or_else(|| store.current_epoch()),
        None => epoch.unwrap_or(EpochId::PENDING),
    };
    if let Some(tx) = tx {
        if types.is_empty() {
            store.record_lpg_dataset_read(tx);
        }
        for ty in types {
            store.record_rel_type_predicate_read(tx, ty);
        }
    }
    if !triangle_node_visible(store, source, epoch, tx) {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    match tx {
        Some(tx) => pairs.extend(store.edges_from_versioned(source, direction, epoch, tx)),
        None if epoch != EpochId::PENDING => {
            store.fill_edges_from_at_epoch(source, direction, epoch, &mut pairs);
        }
        None => store.fill_edges_from(source, direction, &mut pairs),
    }
    pairs.retain(|(node, edge)| {
        // These metadata checks also record actual node/edge reads for SSI.
        let visible = match tx {
            Some(tx) => store.is_edge_visible_versioned(*edge, epoch, tx),
            None => store.is_edge_visible_at_epoch(*edge, epoch),
        };
        if !visible || !triangle_node_visible(store, *node, epoch, tx) {
            return false;
        }
        if types.is_empty() {
            return true;
        }
        // Edge types are immutable. Wrapper defaults can lack the historical or
        // pending type, so retain a versioned fallback only when metadata misses.
        let ty = match tx {
            Some(tx) => store.edge_type_versioned(*edge, epoch, tx).or_else(|| {
                store
                    .get_edge_versioned(*edge, epoch, tx)
                    .map(|edge| edge.edge_type)
            }),
            None => store.edge_type(*edge).or_else(|| {
                store
                    .get_edge_at_epoch(*edge, epoch)
                    .map(|edge| edge.edge_type)
            }),
        };
        ty.is_some_and(|ty| {
            types
                .iter()
                .any(|requested| ty.as_str().eq_ignore_ascii_case(requested))
        })
    });
    pairs
}

/// Columns and adjacency sides for a leapfrog closing hop.
pub struct LeapfrogExpandSpec {
    /// Node column whose adjacency is the left run.
    pub left_column: usize,
    /// Node column whose adjacency is the right run.
    pub right_column: usize,
    /// Traversal direction for the left run.
    pub left_direction: Direction,
    /// Traversal direction for the right run.
    pub right_direction: Direction,
    /// Edge-type filter for the left run (empty = all).
    pub left_edge_types: Vec<String>,
    /// Edge-type filter for the right run (empty = all).
    pub right_edge_types: Vec<String>,
}

/// Closing hop of a directed cycle: for each input row, emit
/// `c ∈ N(left) ∩ N(right)` with the witnessing edges.
///
/// Triangle `(a)-[]->(b)-[]->(c)-[]->(a)`: input is `(a, b)` after the first
/// expand; `left` is `b` outgoing, `right` is `a` incoming.
pub struct LeapfrogExpandOperator {
    store: Arc<dyn GraphStoreSearch>,
    input: Box<dyn Operator>,
    left_column: usize,
    right_column: usize,
    left_direction: Direction,
    right_direction: Direction,
    left_edge_types: Vec<String>,
    right_edge_types: Vec<String>,
    chunk_capacity: usize,
    current_input: Option<DataChunk>,
    current_row: usize,
    pending: Option<IntersectionRows>,
    exhausted: bool,
    transaction_id: Option<TransactionId>,
    viewing_epoch: Option<EpochId>,
    read_only: bool,
    left_cache: FxHashMap<NodeId, Arc<NeighborRun>>,
    right_cache: FxHashMap<NodeId, Arc<NeighborRun>>,
}

impl LeapfrogExpandOperator {
    /// Creates a closing-hop leapfrog expand.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        spec: LeapfrogExpandSpec,
    ) -> Self {
        Self {
            store,
            input,
            left_column: spec.left_column,
            right_column: spec.right_column,
            left_direction: spec.left_direction,
            right_direction: spec.right_direction,
            left_edge_types: spec.left_edge_types,
            right_edge_types: spec.right_edge_types,
            chunk_capacity: 2048,
            current_input: None,
            current_row: 0,
            pending: None,
            exhausted: false,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            left_cache: FxHashMap::default(),
            right_cache: FxHashMap::default(),
        }
    }

    /// Directed triangle close: `c ∈ out(b) ∩ in(a)`.
    #[must_use]
    pub fn directed_triangle_close(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        a_column: usize,
        b_column: usize,
        edge_types: Vec<String>,
    ) -> Self {
        Self::new(
            store,
            input,
            LeapfrogExpandSpec {
                left_column: b_column,
                right_column: a_column,
                left_direction: Direction::Outgoing,
                right_direction: Direction::Incoming,
                left_edge_types: edge_types.clone(),
                right_edge_types: edge_types,
            },
        )
    }

    /// Sets transaction / as-of visibility.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Skip versioned MVCC walks when the query has no pending writes.
    #[must_use]
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    fn load_run(&self, source: NodeId, left: bool) -> NeighborRun {
        let (direction, types) = if left {
            (self.left_direction, self.left_edge_types.as_slice())
        } else {
            (self.right_direction, self.right_edge_types.as_slice())
        };
        let mut pairs = triangle_neighbors(
            self.store.as_ref(),
            source,
            direction,
            types,
            self.viewing_epoch,
            self.transaction_id,
        );
        pairs.sort_unstable_by_key(|(n, e)| (*n, *e));
        let mut run = NeighborRun::default();
        run.nodes.reserve(pairs.len());
        run.edges.reserve(pairs.len());
        for (n, e) in pairs {
            run.nodes.push(n);
            run.edges.push(e);
        }
        run
    }

    fn cached_run(&mut self, source: NodeId, left: bool) -> Arc<NeighborRun> {
        let hit = if left {
            self.left_cache.get(&source).cloned()
        } else {
            self.right_cache.get(&source).cloned()
        };
        if let Some(run) = hit {
            return run;
        }
        let run = Arc::new(self.load_run(source, left));
        if left {
            self.left_cache.insert(source, Arc::clone(&run));
        } else {
            self.right_cache.insert(source, Arc::clone(&run));
        }
        run
    }

    fn intersect_row(&mut self, left_src: NodeId, right_src: NodeId) {
        let left = self.cached_run(left_src, true);
        let right = self.cached_run(right_src, false);
        self.pending = Some(IntersectionRows::new(left, right));
    }

    fn load_next_input(&mut self) -> Result<bool, OperatorError> {
        match self.input.next() {
            Ok(Some(mut chunk)) => {
                chunk.flatten();
                self.current_input = Some(chunk);
                self.current_row = 0;
                self.pending = None;
                Ok(true)
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    fn load_intersection_for_current_row(&mut self) -> Result<bool, OperatorError> {
        let Some(chunk) = &self.current_input else {
            return Ok(false);
        };
        if self.current_row >= chunk.row_count() {
            return Ok(false);
        }
        let left_col = chunk.column(self.left_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.left_column))
        })?;
        let right_col = chunk.column(self.right_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.right_column))
        })?;
        let left_src = left_col
            .get_node_id(self.current_row)
            .ok_or_else(|| OperatorError::Execution("Expected node ID in left column".into()))?;
        let right_src = right_col
            .get_node_id(self.current_row)
            .ok_or_else(|| OperatorError::Execution("Expected node ID in right column".into()))?;
        self.intersect_row(left_src, right_src);
        Ok(true)
    }
}

impl Operator for LeapfrogExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }
        if self.current_input.is_none() {
            if !self.load_next_input()? {
                return Ok(None);
            }
            self.load_intersection_for_current_row()?;
        }
        let input_chunk = self.current_input.as_ref().expect("input loaded");
        let input_col_count = input_chunk.column_count();
        let mut schema: Vec<LogicalType> = (0..input_col_count)
            .map(|i| {
                input_chunk
                    .column(i)
                    .map_or(LogicalType::Any, |c| c.data_type().clone())
            })
            .collect();
        schema.push(LogicalType::Edge);
        schema.push(LogicalType::Node);
        schema.push(LogicalType::Edge);

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);
        let mut count = 0;

        while count < self.chunk_capacity {
            if self.current_input.is_none() {
                if !self.load_next_input()? {
                    break;
                }
                self.load_intersection_for_current_row()?;
            }

            let Some((c, e_left, e_right)) = self.pending.as_mut().and_then(IntersectionRows::next)
            else {
                self.current_row += 1;
                if self.current_row >= self.current_input.as_ref().map_or(0, |c| c.row_count()) {
                    self.current_input = None;
                } else {
                    self.load_intersection_for_current_row()?;
                }
                continue;
            };

            let input = self.current_input.as_ref().expect("input loaded");
            for col_idx in 0..input_col_count {
                if let (Some(src), Some(dst)) = (input.column(col_idx), chunk.column_mut(col_idx))
                    && let Some(value) = src.get_value(self.current_row)
                {
                    dst.push_value(value);
                }
            }
            if let Some(col) = chunk.column_mut(input_col_count) {
                col.push_edge_id(e_left);
            }
            if let Some(col) = chunk.column_mut(input_col_count + 1) {
                col.push_node_id(c);
            }
            if let Some(col) = chunk.column_mut(input_col_count + 2) {
                col.push_edge_id(e_right);
            }
            count += 1;
        }

        if count == 0 {
            return Ok(None);
        }
        chunk.set_count(count);
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.input.reset();
        self.current_input = None;
        self.current_row = 0;
        self.pending = None;
        self.exhausted = false;
        self.left_cache.clear();
        self.right_cache.clear();
    }

    fn name(&self) -> &'static str {
        "LeapfrogExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Directed triangles `(a)→(b)→(c)→(a)` via nested-loop expand (baseline).
pub fn count_nested_loop_triangles(store: &dyn GraphStoreSearch) -> usize {
    let mut n = 0usize;
    for a in store.node_ids() {
        for (b, _) in store.edges_from(a, Direction::Outgoing) {
            for (c, _) in store.edges_from(b, Direction::Outgoing) {
                if store
                    .edges_from(c, Direction::Outgoing)
                    .iter()
                    .any(|(t, _)| *t == a)
                {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Directed triangles via leapfrog intersection `out(b) ∩ in(a)`.
pub fn count_leapfrog_triangles(store: &dyn GraphStoreSearch) -> usize {
    let mut n = 0usize;
    for a in store.node_ids() {
        let mut in_a: Vec<NodeId> = store
            .edges_from(a, Direction::Incoming)
            .into_iter()
            .map(|(nid, _)| nid)
            .collect();
        in_a.sort_unstable();
        in_a.dedup();
        for (b, _) in store.edges_from(a, Direction::Outgoing) {
            let mut out_b: Vec<NodeId> = store
                .edges_from(b, Direction::Outgoing)
                .into_iter()
                .map(|(nid, _)| nid)
                .collect();
            out_b.sort_unstable();
            out_b.dedup();
            n += intersect_count(&out_b, &in_a);
        }
    }
    n
}

/// COUNT(*) kernel for a directed outgoing triangle on a start-node scan.
///
/// For each `a` in `starts`, adds `|out(b) ∩ in(a)|` over `b ∈ out(a)`.
/// Equal-target run lengths preserve parallel-edge multiplicity, matching
/// `(a)→(b)→(c)→(a)`.
pub fn count_directed_triangles(
    store: &dyn GraphStoreSearch,
    starts: &[NodeId],
    edge_types: &[String],
    dest_label: Option<&str>,
    epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    use_versioned: bool,
) -> u64 {
    if triangle_native_eligible(store, edge_types, epoch, transaction_id, use_versioned)
        && let Some(n) = store.try_count_directed_triangles(starts, dest_label)
    {
        return n;
    }
    if let (Some(tx), Some(label)) = (transaction_id, dest_label) {
        store.record_label_predicate_read(tx, label);
    }
    let mut n = 0u64;
    let mut out_cache: FxHashMap<NodeId, Arc<Vec<NodeId>>> = FxHashMap::default();
    let mut in_cache: FxHashMap<NodeId, Arc<Vec<NodeId>>> = FxHashMap::default();
    let dests = |cache: &mut FxHashMap<NodeId, Arc<Vec<NodeId>>>,
                 src: NodeId,
                 dir: Direction|
     -> Arc<Vec<NodeId>> {
        if let Some(hit) = cache.get(&src) {
            return Arc::clone(hit);
        }
        let mut nodes: Vec<_> =
            triangle_neighbors(store, src, dir, edge_types, epoch, transaction_id)
                .into_iter()
                .map(|(node, _)| node)
                .collect();
        if let Some(label) = dest_label {
            nodes.retain(|node| {
                let view = match transaction_id {
                    Some(_) => epoch
                        .filter(|ep| *ep != EpochId::PENDING)
                        .unwrap_or_else(|| store.current_epoch()),
                    None => epoch.unwrap_or(EpochId::PENDING),
                };
                match transaction_id {
                    Some(tx) => store.node_has_label_at_epoch(*node, label, view, tx),
                    None if view == EpochId::PENDING => store.node_has_label(*node, label),
                    None => store
                        .get_node_at_epoch(*node, view)
                        .is_some_and(|node| node.has_label(label)),
                }
            });
        }
        nodes.sort_unstable();
        let run = Arc::new(nodes);
        cache.insert(src, Arc::clone(&run));
        run
    };
    for &a in starts {
        let in_a = dests(&mut in_cache, a, Direction::Incoming);
        let out_a = dests(&mut out_cache, a, Direction::Outgoing);
        for &b in out_a.iter() {
            let out_b = dests(&mut out_cache, b, Direction::Outgoing);
            n += intersect_multiplicity(&out_b, &in_a);
        }
    }
    n
}

fn triangle_native_eligible(
    store: &dyn GraphStoreSearch,
    types: &[String],
    epoch: Option<EpochId>,
    tx: Option<TransactionId>,
    use_versioned: bool,
) -> bool {
    !use_versioned
        && tx.is_none()
        // A numeric epoch is a snapshot even when it equals a store's
        // synthetic current_epoch (CompactStore reports 1).
        && epoch.is_none_or(|ep| ep == EpochId::PENDING)
        && !store.may_have_unresolved_transport_edges()
        && (types.is_empty() || store.all_edges_have_types(types))
}

/// Equal destination runs contribute their Cartesian-product cardinality.
fn intersect_multiplicity(left: &[NodeId], right: &[NodeId]) -> u64 {
    let (mut i, mut j, mut count) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => {
                let target = left[i];
                let a = left[i..].partition_point(|n| *n == target);
                let b = right[j..].partition_point(|n| *n == target);
                count += a as u64 * b as u64;
                i += a;
                j += b;
            }
            std::cmp::Ordering::Less => i = gallop_seek(left, i + 1, &right[j]),
            std::cmp::Ordering::Greater => j = gallop_seek(right, j + 1, &left[i]),
        }
    }
    count
}

/// One-row COUNT(*) of directed outgoing triangles from a start-node scan.
pub struct TriangleCountOperator {
    store: Arc<dyn GraphStoreSearch>,
    input: Box<dyn Operator>,
    edge_types: Vec<String>,
    dest_label: Option<String>,
    count_all: bool,
    transaction_id: Option<TransactionId>,
    viewing_epoch: Option<EpochId>,
    read_only: bool,
    done: bool,
}

impl TriangleCountOperator {
    /// Counts `(a)→(b)→(c)→(a)` for each start node in `input`.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        edge_types: Vec<String>,
    ) -> Self {
        Self {
            store,
            input,
            edge_types,
            dest_label: None,
            count_all: false,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            done: false,
        }
    }

    /// Keep only dests that have this node label (`(b:L)` / `(c:L)`).
    #[must_use]
    pub fn with_dest_label(mut self, label: Option<String>) -> Self {
        self.dest_label = label;
        self
    }

    /// COUNT every start node in the store (skip pulling the scan).
    #[must_use]
    pub fn with_count_all(mut self, count_all: bool) -> Self {
        self.count_all = count_all;
        self
    }

    /// Sets as-of / MVCC visibility.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Skip versioned walks when the query is read-only.
    #[must_use]
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    fn emit_count(n: u64) -> OperatorResult {
        // reason: COUNT(*) fits i64 for this kernel
        #[allow(clippy::cast_possible_wrap)]
        let n_i = n as i64;
        let mut col = crate::execution::vector::ValueVector::with_type(LogicalType::Int64);
        col.push_int64(n_i);
        let mut out = DataChunk::new(vec![col]);
        out.set_count(1);
        Ok(Some(out))
    }
}

impl Operator for TriangleCountOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        if self.count_all
            && triangle_native_eligible(
                self.store.as_ref(),
                &self.edge_types,
                self.viewing_epoch,
                self.transaction_id,
                !self.read_only,
            )
            && let Some(n) = self
                .store
                .try_count_all_directed_triangles(self.dest_label.as_deref())
        {
            return Self::emit_count(n);
        }
        let mut starts = Vec::new();
        // The planned scan owns start visibility and predicate/SSI reads, including
        // historical labels and a transaction's pending nodes.
        while let Some(chunk) = self.input.next()? {
            let Some(col) = chunk.column(0) else {
                continue;
            };
            for i in chunk.selected_indices() {
                if let Some(id) = col.get_node_id(i) {
                    starts.push(id);
                }
            }
        }
        let n = count_directed_triangles(
            self.store.as_ref(),
            &starts,
            &self.edge_types,
            self.dest_label.as_deref(),
            self.viewing_epoch,
            self.transaction_id,
            !self.read_only,
        );
        Self::emit_count(n)
    }

    fn reset(&mut self) {
        self.input.reset();
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "TriangleCount"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "lpg")]
    use super::super::{ExpandOperator, ScanOperator};
    use super::*;
    #[cfg(feature = "compact-store")]
    use crate::graph::compact::csr::CsrAdjacency;
    #[cfg(feature = "lpg")]
    use crate::graph::lpg::LpgStore;

    #[cfg(feature = "lpg")]
    fn triangle_store() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().unwrap());
        let n0 = store.create_node(&["V"]);
        let n1 = store.create_node(&["V"]);
        let n2 = store.create_node(&["V"]);
        let n3 = store.create_node(&["V"]);
        // Directed cycle 0→1→2→0 (3 enumerations, one per start).
        store.create_edge(n0, n1, "R");
        store.create_edge(n1, n2, "R");
        store.create_edge(n2, n0, "R");
        // Second triangle 0→1→3→0.
        store.create_edge(n1, n3, "R");
        store.create_edge(n3, n0, "R");
        // Dead-end, no extra triangle.
        let n4 = store.create_node(&["V"]);
        store.create_edge(n0, n4, "R");
        store
    }

    #[cfg(feature = "lpg")]
    fn parallel_triangle_store() -> (Arc<LpgStore>, [NodeId; 3], [Vec<EdgeId>; 3]) {
        let store = Arc::new(LpgStore::new().unwrap());
        let nodes = [
            store.create_node(&["Start"]),
            store.create_node(&["V"]),
            store.create_node(&["V"]),
        ];
        let edges = std::array::from_fn(|side| {
            (0..side + 2)
                .map(|_| store.create_edge(nodes[side], nodes[(side + 1) % 3], "R"))
                .collect()
        });
        (store, nodes, edges)
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_parallel_rows_cartesian_chunk_one_reset() {
        let (store, nodes, edges) = parallel_triangle_store();
        let store: Arc<dyn GraphStoreSearch> = store;
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Start"));
        let first = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".into()],
        ));
        let mut op =
            LeapfrogExpandOperator::directed_triangle_close(store, first, 0, 2, vec!["R".into()])
                .with_read_only(true);
        op.chunk_capacity = 1;
        let mut expected = Vec::new();
        for &ab in &edges[0] {
            for &bc in &edges[1] {
                for &ca in &edges[2] {
                    expected.push((ab, bc, ca));
                }
            }
        }
        expected.sort_unstable();
        for _ in 0..2 {
            let mut actual = Vec::new();
            while let Some(chunk) = op.next().unwrap() {
                assert_eq!(chunk.row_count(), 1);
                assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(nodes[0]));
                assert_eq!(chunk.column(4).unwrap().get_node_id(0), Some(nodes[2]));
                actual.push((
                    chunk.column(1).unwrap().get_edge_id(0).unwrap(),
                    chunk.column(3).unwrap().get_edge_id(0).unwrap(),
                    chunk.column(5).unwrap().get_edge_id(0).unwrap(),
                ));
            }
            actual.sort_unstable();
            assert_eq!(actual, expected);
            op.reset();
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_parallel_multiple_destination_groups_chunk_one_reset() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&["V"]);
        let d1 = store.create_node(&["V"]);
        let dead = store.create_node(&["V"]);
        let d2 = store.create_node(&["V"]);
        let orphan = store.create_node(&["V"]);
        store.create_edge(a, b, "R");

        // Two matching destination groups, with a non-match between them in
        // each sorted run. Each group must retain its parallel-edge product.
        let left_d1: Vec<_> = (0..2).map(|_| store.create_edge(b, d1, "R")).collect();
        let _left_dead = store.create_edge(b, dead, "R");
        let left_d2: Vec<_> = (0..3).map(|_| store.create_edge(b, d2, "R")).collect();
        let right_d1: Vec<_> = (0..3).map(|_| store.create_edge(d1, a, "R")).collect();
        let _right_orphan = store.create_edge(orphan, a, "R");
        let right_d2: Vec<_> = (0..2).map(|_| store.create_edge(d2, a, "R")).collect();

        let store: Arc<dyn GraphStoreSearch> = store;
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Start"));
        let first = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".into()],
        ));
        let mut op = LeapfrogExpandOperator::directed_triangle_close(
            Arc::clone(&store),
            first,
            0,
            2,
            vec!["R".into()],
        )
        .with_read_only(true);
        op.chunk_capacity = 1;

        let mut expected = Vec::new();
        for &left in &left_d1 {
            for &right in &right_d1 {
                expected.push((left, right));
            }
        }
        for &left in &left_d2 {
            for &right in &right_d2 {
                expected.push((left, right));
            }
        }
        expected.sort_unstable();

        for _ in 0..2 {
            let mut actual = Vec::new();
            while let Some(chunk) = op.next().unwrap() {
                assert_eq!(chunk.row_count(), 1);
                actual.push((
                    chunk.column(3).unwrap().get_edge_id(0).unwrap(),
                    chunk.column(5).unwrap().get_edge_id(0).unwrap(),
                ));
            }
            actual.sort_unstable();
            assert_eq!(actual, expected);
            op.reset();
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_parallel_count_products_and_missing_type() {
        let (store, nodes, _) = parallel_triangle_store();
        for versioned in [false, true] {
            assert_eq!(
                count_directed_triangles(
                    store.as_ref(),
                    &nodes[..1],
                    &["R".into()],
                    None,
                    None,
                    None,
                    versioned
                ),
                24
            );
            assert_eq!(
                count_directed_triangles(
                    store.as_ref(),
                    &nodes,
                    &["R".into()],
                    None,
                    None,
                    None,
                    versioned
                ),
                72
            );
            assert_eq!(
                count_directed_triangles(
                    store.as_ref(),
                    &nodes,
                    &["MISSING".into()],
                    None,
                    None,
                    None,
                    versioned
                ),
                0
            );
        }
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_count_historical_and_own_transaction() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.set_epoch(EpochId::new(1));
        let a = store.create_node(&["Start"]);
        let b = store.create_node(&["V"]);
        let c = store.create_node(&["V"]);
        store.create_edge(a, b, "R");
        store.create_edge(b, c, "R");
        store.create_edge(c, a, "R");
        store.set_epoch(EpochId::new(2));
        store.create_edge(c, a, "R");
        store.create_edge(a, b, "OTHER");
        let tx = TransactionId::new(91);
        store.create_edge_versioned(c, a, "R", EpochId::new(2), tx);
        assert_eq!(
            count_directed_triangles(
                store.as_ref(),
                &[a],
                &["R".into()],
                None,
                Some(EpochId::new(1)),
                None,
                false
            ),
            1
        );
        assert_eq!(
            count_directed_triangles(
                store.as_ref(),
                &[a],
                &["R".into()],
                None,
                Some(EpochId::new(2)),
                Some(tx),
                true
            ),
            3
        );
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_transaction_records_nodes_edges_and_type_predicate() {
        use crate::execution::operators::{ReadTracker, SharedReadTracker};
        use parking_lot::Mutex;
        use std::collections::BTreeSet;
        #[derive(Default)]
        struct Reads {
            nodes: Mutex<BTreeSet<NodeId>>,
            edges: Mutex<BTreeSet<EdgeId>>,
            types: Mutex<BTreeSet<String>>,
        }
        impl ReadTracker for Reads {
            fn record_node_read(&self, _: TransactionId, id: NodeId) {
                self.nodes.lock().insert(id);
            }
            fn record_edge_read(&self, _: TransactionId, id: EdgeId) {
                self.edges.lock().insert(id);
            }
            fn record_rel_type_name_predicate_read(&self, _: TransactionId, ty: &str) {
                self.types.lock().insert(ty.to_string());
            }
        }
        let (store, nodes, edges) = parallel_triangle_store();
        let tx = TransactionId::new(92);
        let spy = Arc::new(Reads::default());
        store.register_read_tracker(tx, Arc::clone(&spy) as SharedReadTracker);
        assert_eq!(
            count_directed_triangles(
                store.as_ref(),
                &nodes[..1],
                &["R".into()],
                None,
                Some(store.current_epoch()),
                Some(tx),
                true
            ),
            24
        );
        assert!(nodes.iter().all(|node| spy.nodes.lock().contains(node)));
        assert!(
            edges
                .iter()
                .flatten()
                .all(|edge| spy.edges.lock().contains(edge))
        );
        assert!(spy.types.lock().contains("R"));
        store.unregister_read_tracker(tx);
    }

    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    #[test]
    fn triangle_count_all_native_requires_requested_type() {
        use crate::graph::compact::builder::CompactStoreBuilder;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(
            CompactStoreBuilder::new()
                .node_table("V", |t| t.column_bitpacked("id", &[0, 1, 2], 2))
                .rel_table("R", "V", "V", |r| {
                    r.edges([(0, 1), (1, 2), (2, 0)]).backward(true)
                })
                .build()
                .unwrap(),
        );
        assert_eq!(store.try_count_all_directed_triangles(None), Some(3));
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let mut op = TriangleCountOperator::new(store, scan, vec!["ABSENT".into()])
            .with_count_all(true)
            .with_read_only(true);
        assert_eq!(
            op.next().unwrap().unwrap().column(0).unwrap().get_int64(0),
            Some(0)
        );
    }

    #[test]
    fn gallop_and_intersect_sorted() {
        let a = [1u32, 3, 5, 7, 9, 11, 13];
        let b = [2, 5, 8, 11, 14];
        assert_eq!(gallop_seek(&a, 0, &5), 2);
        assert_eq!(gallop_seek(&a, 0, &1), 0);
        assert_eq!(gallop_seek(&a, 0, &14), 7);
        let mut out = Vec::new();
        intersect_sorted(&a, &b, &mut out);
        assert_eq!(out, vec![5, 11]);
        assert_eq!(intersect_count(&a, &b), 2);
        assert_eq!(intersect_count(&a, &[] as &[u32]), 0);
    }

    #[cfg(feature = "compact-store")]
    #[test]
    fn csr_neighbor_runs_intersect() {
        // from_sorted_edges only orders by src; dests in a run may be unsorted.
        let mut edges = vec![(0u32, 3), (0, 1), (0, 7), (1, 7), (1, 1), (1, 3)];
        edges.sort_by_key(|&(s, _)| s);
        let csr = CsrAdjacency::from_sorted_edges(2, &edges);
        let mut left = csr.neighbors(0).to_vec();
        let mut right = csr.neighbors(1).to_vec();
        left.sort_unstable();
        right.sort_unstable();
        let mut out = Vec::new();
        intersect_sorted(&left, &right, &mut out);
        assert_eq!(out, vec![1, 3, 7]);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_leapfrog_matches_nested_loop() {
        let store = triangle_store();
        let nested = count_nested_loop_triangles(store.as_ref());
        let leap = count_leapfrog_triangles(store.as_ref());
        let starts: Vec<NodeId> = store.node_ids();
        let dests = count_directed_triangles(store.as_ref(), &starts, &[], None, None, None, false);
        assert_eq!(nested, leap);
        assert_eq!(nested as u64, dests);
        // 0→1→2→0, 1→2→0→1, 2→0→1→2, 0→1→3→0, 1→3→0→1, 3→0→1→3.
        assert_eq!(nested, 6);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn triangle_count_operator_matches_leapfrog() {
        let store: Arc<dyn GraphStoreSearch> = triangle_store();
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let mut op =
            TriangleCountOperator::new(store, scan, vec!["R".to_string()]).with_read_only(true);
        let chunk = op.next().unwrap().expect("one COUNT row");
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column(0).unwrap().get_int64(0), Some(6));
        assert!(op.next().unwrap().is_none());
        op.reset();
        let again = op.next().unwrap().expect("reset COUNT row");
        assert_eq!(again.column(0).unwrap().get_int64(0), Some(6));
        assert_eq!(op.name(), "TriangleCount");
        let any = Box::new(op).into_any();
        assert!(any.downcast::<TriangleCountOperator>().is_ok());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn leapfrog_expand_operator_matches_nested_expand() {
        let store: Arc<dyn GraphStoreSearch> = triangle_store();

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let hop1 = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let hop2 = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            hop1,
            2,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let hop3 = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            hop2,
            4,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let mut nested_op = hop3;
        let mut nested = 0usize;
        while let Some(chunk) = nested_op.next().unwrap() {
            let a = chunk.column(0).unwrap();
            let a2 = chunk.column(6).unwrap();
            for i in 0..chunk.row_count() {
                if a.get_node_id(i) == a2.get_node_id(i) {
                    nested += 1;
                }
            }
        }

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let hop1 = Box::new(ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            vec!["R".to_string()],
        ));
        let mut leap = LeapfrogExpandOperator::directed_triangle_close(
            store,
            hop1,
            0,
            2,
            vec!["R".to_string()],
        );
        let mut leap_n = 0usize;
        while let Some(chunk) = leap.next().unwrap() {
            leap_n += chunk.row_count();
        }
        assert_eq!(nested, leap_n);
        assert_eq!(nested, 6);
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn leapfrog_expand_into_any() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "V"));
        let op = LeapfrogExpandOperator::directed_triangle_close(store, scan, 0, 0, vec![]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<LeapfrogExpandOperator>().is_ok());
    }
}
