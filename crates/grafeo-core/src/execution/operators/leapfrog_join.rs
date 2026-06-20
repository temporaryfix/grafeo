//! Leapfrog TrieJoin operator for worst-case optimal joins.
//!
//! This operator wraps the `LeapfrogJoin` algorithm from the trie index module
//! to provide efficient multi-way joins for cyclic patterns like triangles.
//!
//! Traditional binary hash joins cascade O(N²) for triangle patterns; leapfrog
//! achieves O(N^1.5) by processing all relations simultaneously.
//!
//! # Global-variable alignment
//!
//! A triangle join R1(a,b), R2(b,c), R3(c,a) has `shared_variables = [a, b, c]`.
//! Each input's trie is built in that global order — so R3's path is [a_val, c_val].
//! The recursive intersection visits global variables 0..N in order; at each
//! variable only the inputs that carry it participate in the leapfrog intersection.

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;
use crate::index::trie::{LeapfrogJoin, TrieIndex, TrieIterator};

/// Row identifier for reconstructing output: (input_index, chunk_index, row_index).
type RowId = (usize, usize, usize);

/// A multi-way join intersection result.
struct JoinResult {
    /// Row identifiers from each input that participated in this match.
    row_ids: Vec<Vec<RowId>>,
}

/// Leapfrog TrieJoin operator for worst-case optimal multi-way joins.
///
/// Uses the leapfrog algorithm to efficiently find intersections across
/// multiple sorted inputs without materializing intermediate Cartesian products.
pub struct LeapfrogJoinOperator {
    /// Input operators (one per relation in the join).
    inputs: Vec<Box<dyn Operator>>,

    /// Global-variable alignment: `shared_var_cols[input_idx][global_var_idx]`
    /// = `Some(col_in_input)` if input participates in that variable, else `None`.
    /// Drives both trie construction (path order) and recursive intersection.
    shared_var_cols: Vec<Vec<Option<usize>>>,

    /// Output schema (combined columns from all inputs).
    output_schema: Vec<LogicalType>,

    /// Mapping from output column index to (input_idx, column_idx).
    output_column_mapping: Vec<(usize, usize)>,

    // === Materialization state ===
    /// Materialized input chunks (built once during first next() call).
    materialized_inputs: Vec<Vec<DataChunk>>,

    /// TrieIndex structures built from materialized inputs.
    tries: Vec<TrieIndex>,

    /// Whether materialization is complete.
    materialized: bool,

    // === Iteration state ===
    /// Pre-computed join results.
    results: Vec<JoinResult>,

    /// Current position in results.
    result_position: usize,

    /// Current expansion position within current result's cross product.
    expansion_indices: Vec<usize>,

    /// Whether iteration is exhausted.
    exhausted: bool,
}

impl LeapfrogJoinOperator {
    /// Creates a new leapfrog join operator.
    ///
    /// # Arguments
    /// * `inputs` - Input operators (one per relation).
    /// * `shared_var_cols` - Global-variable alignment: `[input_idx][global_var_idx]`
    ///   = `Some(col_in_input)` if that input carries that variable, else `None`.
    /// * `output_schema` - Schema of the output columns.
    /// * `output_column_mapping` - Maps output columns to (input_idx, column_idx).
    #[must_use]
    pub fn new(
        inputs: Vec<Box<dyn Operator>>,
        shared_var_cols: Vec<Vec<Option<usize>>>,
        output_schema: Vec<LogicalType>,
        output_column_mapping: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            inputs,
            shared_var_cols,
            output_schema,
            output_column_mapping,
            materialized_inputs: Vec::new(),
            tries: Vec::new(),
            materialized: false,
            results: Vec::new(),
            result_position: 0,
            expansion_indices: Vec::new(),
            exhausted: false,
        }
    }

    /// Materializes all inputs and builds trie indexes in global-variable order.
    fn materialize_inputs(&mut self) -> Result<(), OperatorError> {
        // Phase 1: Collect all chunks from each input
        for input in &mut self.inputs {
            let mut chunks = Vec::new();
            while let Some(chunk) = input.next()? {
                chunks.push(chunk);
            }
            self.materialized_inputs.push(chunks);
        }

        // Phase 2: Build TrieIndex for each input.
        // The trie path for input i is the sequence of column values for each
        // global variable that input i carries, in global-variable order.
        for (input_idx, chunks) in self.materialized_inputs.iter().enumerate() {
            let mut trie = TrieIndex::new();
            // Collect which columns to use, in global-var order (skip None entries)
            let key_indices: Vec<usize> = self.shared_var_cols[input_idx]
                .iter()
                .filter_map(|&opt| opt)
                .collect();

            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                for row in 0..chunk.row_count() {
                    if let Some(path) = Self::extract_join_keys_static(chunk, row, &key_indices) {
                        let row_id = Self::encode_row_id(input_idx, chunk_idx, row);
                        trie.insert(&path, row_id);
                    }
                }
            }
            self.tries.push(trie);
        }

        self.materialized = true;
        Ok(())
    }

    /// Extracts join key values from a row (static version, no &self borrow needed).
    fn extract_join_keys_static(
        chunk: &DataChunk,
        row: usize,
        key_indices: &[usize],
    ) -> Option<Vec<NodeId>> {
        let mut path = Vec::with_capacity(key_indices.len());

        for &col_idx in key_indices {
            let col = chunk.column(col_idx)?;
            let node_id = match col.data_type() {
                LogicalType::Node => col.get_node_id(row),
                LogicalType::Edge => col.get_edge_id(row).map(|e| NodeId::new(e.as_u64())),
                // reason: ID encoding: i64 <-> u64 round-trip
                LogicalType::Int64 => col.get_int64(row).map(|i| {
                    // reason: ID encoding: i64 to u64 round-trip, negative IDs do not occur
                    #[allow(clippy::cast_sign_loss)]
                    NodeId::new(i as u64)
                }),
                _ => return None,
            }?;
            path.push(node_id);
        }

        Some(path)
    }

    /// Encodes a row location as an EdgeId for trie storage.
    fn encode_row_id(input_idx: usize, chunk_idx: usize, row: usize) -> EdgeId {
        // Pack: input (8 bits) | chunk (24 bits) | row (32 bits)
        let encoded = ((input_idx as u64) << 56)
            | ((chunk_idx as u64 & 0xFFFFFF) << 32)
            | (row as u64 & 0xFFFFFFFF);
        EdgeId::new(encoded)
    }

    /// Decodes a row location from an EdgeId.
    fn decode_row_id(edge_id: EdgeId) -> RowId {
        let encoded = edge_id.as_u64();
        let input_idx = (encoded >> 56) as usize;
        let chunk_idx = ((encoded >> 32) & 0xFFFFFF) as usize;
        let row = (encoded & 0xFFFFFFFF) as usize;
        (input_idx, chunk_idx, row)
    }

    /// Executes the recursive multi-level leapfrog join.
    fn execute_leapfrog(&mut self) -> Result<(), OperatorError> {
        if self.tries.is_empty() {
            return Ok(());
        }

        let num_inputs = self.tries.len();
        let num_shared_vars = self.shared_var_cols.first().map_or(0, |v| v.len());

        // Start each input's iterator at the root of its trie
        let initial_iters: Vec<TrieIterator<'_>> = self.tries.iter().map(|t| t.iter()).collect();

        // Collect results via recursion. We can't hold &mut self while borrowing
        // self.tries, so we collect results separately and extend self.results.
        let mut collected: Vec<JoinResult> = Vec::new();
        Self::recurse(
            0,
            num_shared_vars,
            &self.shared_var_cols,
            initial_iters,
            &self.tries,
            &mut collected,
        );

        self.results.extend(collected);

        // Initialize expansion indices if we have results
        if !self.results.is_empty() {
            self.expansion_indices = vec![0; num_inputs];
        }

        Ok(())
    }

    /// Recursive multi-level leapfrog triejoin.
    ///
    /// `gv`: current global variable index (0..num_shared_vars).
    /// `per_input_iters`: for each input, its current `TrieIterator` — positioned
    ///   at the trie level for "the next variable this input participates in".
    ///   Non-participating inputs (for the current gv) keep their iterators
    ///   unchanged across this level.
    ///
    /// # Iterator semantics
    ///
    /// Each input's trie is built in global-variable order (only the vars it carries).
    /// So for R3(c,a) with shared_vars=[a,b,c], R3's trie has levels [a, c].
    /// When processing gv=0 (a), R3's iterator is at root level (iterating a values).
    /// When processing gv=2 (c), R3's iterator is at level 1 (iterating c values given a).
    ///
    /// We use `LeapfrogJoin` to find the intersection key, then manually descend
    /// each participant's individual iterator (kept outside the leapfrog join struct
    /// to preserve the input_idx ↔ iterator correspondence).
    fn recurse<'a>(
        gv: usize,
        num_shared_vars: usize,
        shared_var_cols: &[Vec<Option<usize>>],
        per_input_iters: Vec<TrieIterator<'a>>,
        tries: &'a [TrieIndex],
        collected: &mut Vec<JoinResult>,
    ) {
        if gv == num_shared_vars {
            // All variables bound. Collect row-ids from each input's leaf node.
            // `per_input_iters[i]` is positioned at the leaf node for input i
            // (its node's `edges` SmallVec holds the encoded row-ids).
            let num_inputs = tries.len();
            let mut row_ids_per_input: Vec<Vec<RowId>> = vec![Vec::new(); num_inputs];

            for (input_idx, iter) in per_input_iters.iter().enumerate() {
                Self::collect_leaf_row_ids(iter, input_idx, &mut row_ids_per_input[input_idx]);
            }

            if row_ids_per_input.iter().all(|ids| !ids.is_empty()) {
                collected.push(JoinResult {
                    row_ids: row_ids_per_input,
                });
            }
            return;
        }

        // Find which inputs participate in this global variable
        let participant_indices: Vec<usize> = (0..shared_var_cols.len())
            .filter(|&i| shared_var_cols[i][gv].is_some())
            .collect();

        if participant_indices.is_empty() {
            // No input participates at this variable — skip to next
            Self::recurse(
                gv + 1,
                num_shared_vars,
                shared_var_cols,
                per_input_iters,
                tries,
                collected,
            );
            return;
        }

        // Build a leapfrog join over CLONES of participant iterators.
        // `LeapfrogJoin` sorts its iterators internally, so we cannot reliably
        // recover which sorted slot corresponds to which original input index.
        // Instead, we keep the participant iterators ourselves (in `per_input_iters`,
        // already owned by this stack frame) and use `lj` solely for key enumeration.
        // For each key that `lj` reports, we seek the originals to that key and open.
        let lj_iters: Vec<TrieIterator<'a>> = participant_indices
            .iter()
            .map(|&i| per_input_iters[i].clone())
            .collect();

        let mut lj = LeapfrogJoin::new(lj_iters);

        // We need owned mutable copies of the participant iterators.
        // Take them out of per_input_iters (we'll reconstruct next_iters per key).
        // Use a separate vec so we can mutate (seek) without consuming per_input_iters.
        let mut owned_participant_iters: Vec<TrieIterator<'a>> = participant_indices
            .iter()
            .map(|&i| per_input_iters[i].clone())
            .collect();

        loop {
            let Some(key) = lj.key() else { break };

            // Seek each participant's owned iterator to `key` and open its child.
            // `seek` is forward-only but guarantees landing on `key` because
            // LeapfrogJoin confirmed this key exists in all participants.
            let mut open_ok = true;
            let mut child_iters: Vec<TrieIterator<'a>> =
                Vec::with_capacity(participant_indices.len());
            for pit in &mut owned_participant_iters {
                pit.seek(key);
                match pit.open() {
                    Some(child) => child_iters.push(child),
                    None => {
                        // Should not happen for a valid intersection key
                        open_ok = false;
                        break;
                    }
                }
            }
            if !open_ok {
                break;
            }

            // Build next per_input_iters:
            // - participants: replaced with their child iterator (one level down)
            // - non-participants: cloned unchanged (they don't bind at this gv)
            let mut next_iters: Vec<TrieIterator<'a>> = per_input_iters.clone();
            for (slot, &input_idx) in participant_indices.iter().enumerate() {
                next_iters[input_idx] = child_iters[slot].clone();
            }

            Self::recurse(
                gv + 1,
                num_shared_vars,
                shared_var_cols,
                next_iters,
                tries,
                collected,
            );

            if !lj.next() {
                break;
            }
        }
    }

    /// Collects all EdgeId-encoded row IDs from a TrieIterator's current node
    /// (the leaf node after all descents). The trie stores row-ids in `node.edges`.
    ///
    /// Since `TrieIterator` doesn't expose `node.edges` directly, we use the trie's
    /// `get` method via the iterator's exposed `collect_edges` helper. We expose
    /// this by walking the iterator's current node's edges indirectly: we call
    /// `iter.collect_edges()` which we've added to `TrieIterator`.
    fn collect_leaf_row_ids(iter: &TrieIterator<'_>, input_idx: usize, row_ids: &mut Vec<RowId>) {
        for &edge_id in iter.edges() {
            let decoded = Self::decode_row_id(edge_id);
            if decoded.0 == input_idx {
                row_ids.push(decoded);
            }
        }
    }

    /// Advances to the next combination in the current result's cross product.
    fn advance_expansion(&mut self) -> bool {
        if self.result_position >= self.results.len() {
            return false;
        }

        let result = &self.results[self.result_position];

        // Try to advance from the rightmost input
        for i in (0..self.expansion_indices.len()).rev() {
            self.expansion_indices[i] += 1;
            if self.expansion_indices[i] < result.row_ids[i].len() {
                return true;
            }
            self.expansion_indices[i] = 0;
        }

        // All combinations exhausted for this result, move to next
        self.result_position += 1;
        if self.result_position < self.results.len() {
            self.expansion_indices = vec![0; self.inputs.len()];
            true
        } else {
            false
        }
    }

    /// Builds an output row from the current expansion position.
    fn build_output_row(&self, builder: &mut DataChunkBuilder) -> Result<(), OperatorError> {
        let result = &self.results[self.result_position];

        for (out_col, &(input_idx, in_col)) in self.output_column_mapping.iter().enumerate() {
            let expansion_idx = self.expansion_indices[input_idx];
            let (_, chunk_idx, row) = result.row_ids[input_idx][expansion_idx];

            let chunk = &self.materialized_inputs[input_idx][chunk_idx];
            let col = chunk
                .column(in_col)
                .ok_or_else(|| OperatorError::ColumnNotFound(in_col.to_string()))?;

            let out_col_vec = builder
                .column_mut(out_col)
                .ok_or_else(|| OperatorError::ColumnNotFound(out_col.to_string()))?;

            if let Some(value) = col.get_value(row) {
                out_col_vec.push_value(value);
            } else {
                out_col_vec.push_value(Value::Null);
            }
        }

        builder.advance_row();
        Ok(())
    }
}

impl Operator for LeapfrogJoinOperator {
    fn next(&mut self) -> OperatorResult {
        // First call: materialize inputs and execute leapfrog
        if !self.materialized {
            self.materialize_inputs()?;
            self.execute_leapfrog()?;
        }

        if self.exhausted || self.results.is_empty() {
            return Ok(None);
        }

        // Check if we've exhausted all results
        if self.result_position >= self.results.len() {
            self.exhausted = true;
            return Ok(None);
        }

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        while !builder.is_full() {
            self.build_output_row(&mut builder)?;

            if !self.advance_expansion() {
                self.exhausted = true;
                break;
            }
        }

        if builder.row_count() > 0 {
            Ok(Some(builder.finish()))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        for input in &mut self.inputs {
            input.reset();
        }
        self.materialized_inputs.clear();
        self.tries.clear();
        self.materialized = false;
        self.results.clear();
        self.result_position = 0;
        self.expansion_indices.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "LeapfrogJoin"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ── helpers for tests ──────────────────────────────────────────────────────────

/// Builds a `shared_var_cols` alignment for the common case where each input
/// shares exactly 1 variable at column 0.  Used by the old-style single-var tests.
///
/// `n_inputs` inputs, each participating in global variable 0, at column 0.
#[cfg(test)]
fn single_var_alignment(n_inputs: usize) -> Vec<Vec<Option<usize>>> {
    vec![vec![Some(0)]; n_inputs]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;

    /// Creates a simple scan operator that returns a single chunk.
    struct MockScanOperator {
        chunk: Option<DataChunk>,
        returned: bool,
    }

    impl MockScanOperator {
        fn new(chunk: DataChunk) -> Self {
            Self {
                chunk: Some(chunk),
                returned: false,
            }
        }
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.returned {
                Ok(None)
            } else {
                self.returned = true;
                Ok(self.chunk.take())
            }
        }

        fn reset(&mut self) {
            self.returned = false;
        }

        fn name(&self) -> &'static str {
            "MockScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn create_node_chunk(node_ids: &[i64]) -> DataChunk {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        for &id in node_ids {
            col.push_int64(id);
        }
        DataChunk::new(vec![col])
    }

    /// Creates a 2-column chunk from pairs of i64 values.
    fn create_two_col_chunk(pairs: &[(i64, i64)]) -> DataChunk {
        let mut col0 = ValueVector::with_type(LogicalType::Int64);
        let mut col1 = ValueVector::with_type(LogicalType::Int64);
        for &(a, b) in pairs {
            col0.push_int64(a);
            col1.push_int64(b);
        }
        DataChunk::new(vec![col0, col1])
    }

    // ── existing single-variable tests (must keep passing) ───────────────────

    #[test]
    fn test_leapfrog_binary_intersection() {
        // Input 1: nodes [1, 2, 3, 5]
        // Input 2: nodes [2, 3, 4, 5]
        // Expected intersection: [2, 3, 5]

        let chunk1 = create_node_chunk(&[1, 2, 3, 5]);
        let chunk2 = create_node_chunk(&[2, 3, 4, 5]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            single_var_alignment(2),
            vec![LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (1, 0)],
        );

        let mut all_results = Vec::new();
        while let Some(chunk) = leapfrog.next().unwrap() {
            for row in 0..chunk.row_count() {
                let val1 = chunk.column(0).unwrap().get_int64(row).unwrap();
                let val2 = chunk.column(1).unwrap().get_int64(row).unwrap();
                all_results.push((val1, val2));
            }
        }

        // Should find 3 matches: (2,2), (3,3), (5,5)
        assert_eq!(all_results.len(), 3);
        assert!(all_results.contains(&(2, 2)));
        assert!(all_results.contains(&(3, 3)));
        assert!(all_results.contains(&(5, 5)));
    }

    #[test]
    fn test_leapfrog_empty_intersection() {
        // Input 1: nodes [1, 2, 3]
        // Input 2: nodes [4, 5, 6]
        // Expected: empty

        let chunk1 = create_node_chunk(&[1, 2, 3]);
        let chunk2 = create_node_chunk(&[4, 5, 6]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            single_var_alignment(2),
            vec![LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (1, 0)],
        );

        let result = leapfrog.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_leapfrog_reset() {
        let chunk1 = create_node_chunk(&[1, 2, 3]);
        let chunk2 = create_node_chunk(&[2, 3, 4]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1.clone()));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2.clone()));

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            single_var_alignment(2),
            vec![LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (1, 0)],
        );

        // First iteration - consume all results
        let mut _count = 0;
        while leapfrog.next().unwrap().is_some() {
            _count += 1;
        }

        // Reset won't work with MockScanOperator since the chunk is taken
        // but the reset logic itself should work
        leapfrog.reset();
        assert!(!leapfrog.materialized);
        assert!(leapfrog.results.is_empty());
    }

    #[test]
    fn test_encode_decode_row_id() {
        let test_cases = [
            (0, 0, 0),
            (1, 2, 3),
            (255, 16777215, 4294967295), // Max values for each field
        ];

        for (input_idx, chunk_idx, row) in test_cases {
            let encoded = LeapfrogJoinOperator::encode_row_id(input_idx, chunk_idx, row);
            let decoded = LeapfrogJoinOperator::decode_row_id(encoded);
            assert_eq!(decoded, (input_idx, chunk_idx, row));
        }
    }

    // ── NEW test 1: aligned 2-column join must intersect BOTH variables ───────

    /// Two inputs share 2 join variables.  The old (broken) code returned the
    /// Cartesian product at the first variable level; the fixed code must return
    /// only the rows where BOTH variables agree.
    ///
    /// Input1 rows (a,b): (1,1), (1,2)
    /// Input2 rows (a,b): (1,1), (1,3)
    ///
    /// Only (1,1)×(1,1) matches — 1 output row (not 4).
    #[test]
    fn aligned_two_column_join_intersects_both() {
        // shared_variables = [a, b]; both inputs carry both vars at cols [0, 1]
        // shared_var_cols[0] = [Some(0), Some(1)]
        // shared_var_cols[1] = [Some(0), Some(1)]
        let chunk1 = create_two_col_chunk(&[(1, 1), (1, 2)]);
        let chunk2 = create_two_col_chunk(&[(1, 1), (1, 3)]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

        let shared_var_cols = vec![
            vec![Some(0), Some(1)], // input 0: col 0 = gv0(a), col 1 = gv1(b)
            vec![Some(0), Some(1)], // input 1: col 0 = gv0(a), col 1 = gv1(b)
        ];

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            shared_var_cols,
            vec![
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Int64,
            ],
            vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        );

        let mut all_rows: Vec<(i64, i64, i64, i64)> = Vec::new();
        while let Some(chunk) = leapfrog.next().unwrap() {
            for row in 0..chunk.row_count() {
                let a1 = chunk.column(0).unwrap().get_int64(row).unwrap();
                let b1 = chunk.column(1).unwrap().get_int64(row).unwrap();
                let a2 = chunk.column(2).unwrap().get_int64(row).unwrap();
                let b2 = chunk.column(3).unwrap().get_int64(row).unwrap();
                all_rows.push((a1, b1, a2, b2));
            }
        }

        // Must be exactly 1 row: (1,1,1,1)
        assert_eq!(
            all_rows.len(),
            1,
            "Expected exactly 1 match (both vars agree), got {}: {:?}",
            all_rows.len(),
            all_rows
        );
        assert_eq!(all_rows[0], (1, 1, 1, 1));
    }

    // ── NEW test 2: triangle ragged join ─────────────────────────────────────

    /// Triangle join with ragged variable sets (the crux of the original bug).
    ///
    /// Relations (encoding via two-column Int64 chunks):
    ///   R1(a, b): col0=a, col1=b
    ///   R2(b, c): col0=b, col1=c
    ///   R3(c, a): col0=c, col1=a
    ///
    /// shared_variables = [a, b, c]
    ///
    ///   R1 → shared_var_cols[0] = [Some(0), Some(1), None]  (a@col0, b@col1)
    ///   R2 → shared_var_cols[1] = [None,    Some(0), Some(1)] (b@col0, c@col1)
    ///   R3 → shared_var_cols[2] = [Some(1), None,   Some(0)] (a@col1, c@col0)
    ///
    /// Trie build order (global-var order per input):
    ///   R1 trie path = [a_val, b_val]
    ///   R2 trie path = [b_val, c_val]
    ///   R3 trie path = [a_val, c_val]   ← descends by a first (gv=0), then c (gv=2)
    ///
    /// Data — exactly one closing triangle: a=1, b=2, c=3
    ///
    ///   R1: (1,2), (1,99)          — decoy (1,99): a=1 matches but b=99 has no partner in R2
    ///   R2: (2,3), (2,88), (99,50) — decoy (2,88): b=2 matches but c=88 has no partner in R3
    ///                               — decoy (99,50): b=99 from R1-decoy has no c-partner in R3
    ///   R3: (3,1), (3,77)          — decoy (3,77): c=3 matches but a=77 not in R1
    ///
    /// Correct result: exactly 1 row — the triangle (a=1,b=2,c=3).
    #[test]
    fn triangle_ragged_join_correct() {
        // R1(a,b): the closing edge a=1→b=2, plus a decoy with b=99
        let chunk_r1 = create_two_col_chunk(&[(1, 2), (1, 99)]);
        // R2(b,c): the closing edge b=2→c=3, plus decoys
        let chunk_r2 = create_two_col_chunk(&[(2, 3), (2, 88), (99, 50)]);
        // R3(c,a): the closing edge c=3→a=1, plus a decoy with different a
        let chunk_r3 = create_two_col_chunk(&[(3, 1), (3, 77)]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk_r1));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk_r2));
        let op3: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk_r3));

        // shared_variables = [a, b, c]
        // R1(a,b): a at col0 (gv=0), b at col1 (gv=1), c absent (gv=2)
        // R2(b,c): a absent (gv=0), b at col0 (gv=1), c at col1 (gv=2)
        // R3(c,a): a at col1 (gv=0), b absent (gv=1), c at col0 (gv=2)
        let shared_var_cols = vec![
            vec![Some(0), Some(1), None], // R1: a@col0, b@col1
            vec![None, Some(0), Some(1)], // R2: b@col0, c@col1
            vec![Some(1), None, Some(0)], // R3: a@col1, c@col0
        ];

        // Output: a, b, c — take from R1(col0=a, col1=b) and R2(col1=c)
        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2, op3],
            shared_var_cols,
            vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (0, 1), (1, 1)], // a from R1.col0, b from R1.col1, c from R2.col1
        );

        let mut all_rows: Vec<(i64, i64, i64)> = Vec::new();
        while let Some(chunk) = leapfrog.next().unwrap() {
            for row in 0..chunk.row_count() {
                let a = chunk.column(0).unwrap().get_int64(row).unwrap();
                let b = chunk.column(1).unwrap().get_int64(row).unwrap();
                let c = chunk.column(2).unwrap().get_int64(row).unwrap();
                all_rows.push((a, b, c));
            }
        }

        assert_eq!(
            all_rows.len(),
            1,
            "Expected exactly 1 triangle (a=1,b=2,c=3), got {}: {:?}",
            all_rows.len(),
            all_rows
        );
        assert_eq!(
            all_rows[0],
            (1, 2, 3),
            "Triangle values must be (a=1,b=2,c=3)"
        );
    }
}
