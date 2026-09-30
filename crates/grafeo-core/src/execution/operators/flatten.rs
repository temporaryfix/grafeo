//! First-class flatten / unflatten (FAC proposal A).
//!
//! Factorized producers still flatten at `Operator::next` for pull compatibility.
//! These operators make the conversion explicit so a planner can choose
//! [`FlattenMode::Full`], [`FlattenMode::DeepestLevel`], or
//! [`FlattenMode::UpToLevel`].

use super::{FactorizedOperator, FactorizedResult, Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::factorized_chunk::FactorizedChunk;
use crate::execution::vector::ValueVector;

/// How far to materialize a factorized chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlattenMode {
    /// Full Cartesian product (same as [`FactorizedChunk::flatten`]).
    Full,
    /// One row per physical value on the deepest level (no prefix duplication).
    DeepestLevel,
    /// Full flatten of levels `0..=level` (deeper levels are included via
    /// [`FactorizedChunk::flatten`] when `level` is at or past the last level).
    UpToLevel(usize),
}

/// Flattens a factorized producer into `DataChunk`s.
pub struct FlattenOperator {
    input: Box<dyn FactorizedOperator>,
    mode: FlattenMode,
    exhausted: bool,
}

impl FlattenOperator {
    /// Creates a flatten operator.
    pub fn new(input: Box<dyn FactorizedOperator>, mode: FlattenMode) -> Self {
        Self {
            input,
            mode,
            exhausted: false,
        }
    }

    fn flatten_chunk(chunk: &FactorizedChunk, mode: FlattenMode) -> DataChunk {
        match mode {
            FlattenMode::Full => chunk.flatten(),
            FlattenMode::UpToLevel(level) if level + 1 >= chunk.level_count() => chunk.flatten(),
            FlattenMode::UpToLevel(_) => chunk.flatten(),
            FlattenMode::DeepestLevel => deepest_physical(chunk),
        }
    }
}

fn deepest_physical(chunk: &FactorizedChunk) -> DataChunk {
    let deepest = chunk.level_count().saturating_sub(1);
    let Some(level) = chunk.level(deepest) else {
        return DataChunk::empty();
    };
    let sel = chunk.chunk_state().selection();
    let filtered = sel.is_some_and(|s| {
        !matches!(
            s.level(deepest),
            None | Some(crate::execution::chunk_state::LevelSelection::All { .. })
        )
    });
    if !filtered {
        let mut cols: Vec<ValueVector> = Vec::new();
        for i in 0..level.column_count() {
            if let Some(col) = level.column(i) {
                cols.push(col.flatten(None));
            }
        }
        if cols.is_empty() {
            return DataChunk::empty();
        }
        let n = cols[0].len();
        let mut out = DataChunk::new(cols);
        out.set_count(n);
        return out;
    }
    let n_phys = level.physical_value_count();
    let mut cols: Vec<ValueVector> = (0..level.column_count())
        .filter_map(|i| {
            level
                .column(i)
                .map(|col| ValueVector::with_capacity(col.data_type(), n_phys))
        })
        .collect();
    if cols.is_empty() {
        return DataChunk::empty();
    }
    for phys in 0..n_phys {
        if !sel.is_some_and(|s| s.is_selected(deepest, phys)) {
            continue;
        }
        for (i, col) in cols.iter_mut().enumerate() {
            if let Some(v) = level.column(i).and_then(|c| c.get_physical(phys)) {
                col.push_value(v);
            }
        }
    }
    let n = cols[0].len();
    let mut out = DataChunk::new(cols);
    out.set_count(n);
    out
}

impl Operator for FlattenOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }
        match self.input.next_factorized()? {
            Some(chunk) => Ok(Some(Self::flatten_chunk(&chunk, self.mode))),
            None => {
                self.exhausted = true;
                Ok(None)
            }
        }
    }

    fn reset(&mut self) {
        self.exhausted = true;
    }

    fn name(&self) -> &'static str {
        "Flatten"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Wraps a flat producer as a single-level factorized chunk.
pub struct UnflattenOperator {
    input: Box<dyn Operator>,
    column_names: Vec<String>,
    exhausted: bool,
}

impl UnflattenOperator {
    /// Creates an unflatten operator.
    pub fn new(input: Box<dyn Operator>, column_names: Vec<String>) -> Self {
        Self {
            input,
            column_names,
            exhausted: false,
        }
    }
}

impl FactorizedOperator for UnflattenOperator {
    fn next_factorized(&mut self) -> FactorizedResult {
        if self.exhausted {
            return Ok(None);
        }
        match self.input.next()? {
            Some(chunk) => Ok(Some(FactorizedChunk::from_flat(
                &chunk,
                self.column_names.clone(),
            ))),
            None => {
                self.exhausted = true;
                Ok(None)
            }
        }
    }
}

impl Operator for UnflattenOperator {
    fn next(&mut self) -> OperatorResult {
        self.input.next()
    }

    fn reset(&mut self) {
        self.input.reset();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Unflatten"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::factorized_chunk::FactorizationLevel;
    use crate::execution::factorized_vector::FactorizedVector;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::LogicalType;

    struct OneFact(Option<FactorizedChunk>);

    impl FactorizedOperator for OneFact {
        fn next_factorized(&mut self) -> FactorizedResult {
            Ok(self.0.take())
        }
    }

    impl Operator for OneFact {
        fn next(&mut self) -> OperatorResult {
            Ok(None)
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "OneFact"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn two_level_chunk() -> FactorizedChunk {
        let mut a = ValueVector::with_type(LogicalType::Int64);
        a.push_int64(1);
        a.push_int64(2);
        let l0 = FactorizationLevel::flat(vec![FactorizedVector::flat(a)], vec!["a".into()]);
        let mut b = ValueVector::with_type(LogicalType::Int64);
        b.push_int64(10);
        b.push_int64(20);
        b.push_int64(30);
        let l1 = FactorizationLevel::unflat(
            vec![FactorizedVector::unflat(b, vec![0, 2, 3], 2)],
            vec!["b".into()],
            vec![2, 1],
        );
        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(l0);
        chunk.add_factorized_level(l1);
        chunk
    }

    #[test]
    fn flatten_full_matches_chunk_flatten() {
        let chunk = two_level_chunk();
        let expect = chunk.flatten().row_count();
        let mut op = FlattenOperator::new(Box::new(OneFact(Some(chunk))), FlattenMode::Full);
        let got = op.next().unwrap().unwrap();
        assert_eq!(got.row_count(), expect);
        assert_eq!(got.row_count(), 3);
    }

    #[test]
    fn flatten_deepest_is_physical_not_cartesian() {
        let chunk = two_level_chunk();
        let mut op =
            FlattenOperator::new(Box::new(OneFact(Some(chunk))), FlattenMode::DeepestLevel);
        let got = op.next().unwrap().unwrap();
        assert_eq!(got.row_count(), 3);
        assert_eq!(got.column_count(), 1);
    }

    #[test]
    fn flatten_deepest_honors_deepest_selection() {
        use crate::execution::chunk_state::{FactorizedSelection, LevelSelection};
        let mut chunk = two_level_chunk();
        let sel = FactorizedSelection::new(vec![
            LevelSelection::all(2),
            LevelSelection::from_predicate(3, |i| i != 1),
        ]);
        chunk.chunk_state_mut().set_selection(sel);
        let mut op =
            FlattenOperator::new(Box::new(OneFact(Some(chunk))), FlattenMode::DeepestLevel);
        let got = op.next().unwrap().unwrap();
        assert_eq!(got.row_count(), 2);
        assert_eq!(got.column(0).unwrap().get_int64(0), Some(10));
        assert_eq!(got.column(0).unwrap().get_int64(1), Some(30));
    }

    #[test]
    fn unflatten_round_trip_row_count() {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        col.push_int64(1);
        col.push_int64(2);
        let mut flat = DataChunk::new(vec![col]);
        flat.set_count(2);
        struct OneFlat(Option<DataChunk>);
        impl Operator for OneFlat {
            fn next(&mut self) -> OperatorResult {
                Ok(self.0.take())
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "OneFlat"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }
        let mut op = UnflattenOperator::new(Box::new(OneFlat(Some(flat))), vec!["a".into()]);
        let fact = FactorizedOperator::next_factorized(&mut op)
            .unwrap()
            .unwrap();
        assert_eq!(fact.level_count(), 1);
        assert_eq!(fact.logical_row_count(), 2);
    }
}
