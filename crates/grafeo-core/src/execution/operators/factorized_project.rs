//! Project selected columns of a factorized chunk without flattening first.
//!
//! Honours `design-proposals-factorization-wcoj` proposal A:
//! `FactorizedChunk::project` is the kernel; this operator is the planner hook.
//! The executor prefers [`FactorizedOperator::next_factorized`] and flattens
//! once when collecting rows. `Operator::next` still flattens for flat callers.

use super::{FactorizedOperator, FactorizedResult, Operator, OperatorResult};

/// `(level, column, output_name)` as accepted by [`crate::execution::FactorizedChunk::project`].
pub type FactorizedColumnSpec = (usize, usize, String);

/// Projects a factorized input, keeping unflat levels until flatten.
pub struct FactorizedProjectOperator {
    input: Box<dyn FactorizedOperator>,
    specs: Vec<FactorizedColumnSpec>,
    exhausted: bool,
}

impl FactorizedProjectOperator {
    /// Creates a project over a factorized producer.
    pub fn new(input: Box<dyn FactorizedOperator>, specs: Vec<FactorizedColumnSpec>) -> Self {
        Self {
            input,
            specs,
            exhausted: false,
        }
    }

    /// Next projected factorized chunk.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input operator fails.
    pub fn next_factorized(&mut self) -> FactorizedResult {
        if self.exhausted {
            return Ok(None);
        }
        match self.input.next_factorized()? {
            Some(chunk) => Ok(Some(chunk.project(&self.specs))),
            None => {
                self.exhausted = true;
                Ok(None)
            }
        }
    }
}

impl FactorizedOperator for FactorizedProjectOperator {
    fn next_factorized(&mut self) -> FactorizedResult {
        FactorizedProjectOperator::next_factorized(self)
    }
}

impl Operator for FactorizedProjectOperator {
    fn next(&mut self) -> OperatorResult {
        match FactorizedProjectOperator::next_factorized(self)? {
            Some(chunk) => Ok(Some(chunk.flatten())),
            None => Ok(None),
        }
    }

    fn reset(&mut self) {
        self.exhausted = true;
    }

    fn name(&self) -> &'static str {
        "FactorizedProject"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn as_factorized_mut(&mut self) -> Option<&mut dyn FactorizedOperator> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::OperatorError;
    use super::*;
    use crate::execution::factorized_chunk::{FactorizationLevel, FactorizedChunk};
    use crate::execution::factorized_vector::FactorizedVector;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::LogicalType;

    struct OneChunk(Option<FactorizedChunk>);

    impl FactorizedOperator for OneChunk {
        fn next_factorized(&mut self) -> FactorizedResult {
            Ok(self.0.take())
        }
    }

    impl Operator for OneChunk {
        fn next(&mut self) -> OperatorResult {
            Err(OperatorError::Execution("use next_factorized".into()))
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "OneChunk"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn project_keeps_unflat_and_logical_count() {
        let mut src = ValueVector::with_type(LogicalType::Int64);
        src.push_int64(1);
        src.push_int64(2);
        let l0 = FactorizationLevel::flat(vec![FactorizedVector::flat(src)], vec!["a".into()]);
        let mut mid = ValueVector::with_type(LogicalType::Int64);
        mid.push_int64(10);
        mid.push_int64(20);
        mid.push_int64(30);
        let l1 = FactorizationLevel::unflat(
            vec![FactorizedVector::unflat(mid, vec![0, 2, 3], 2)],
            vec!["b".into()],
            vec![2, 1],
        );
        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(l0);
        chunk.add_factorized_level(l1);

        let mut op = FactorizedProjectOperator::new(
            Box::new(OneChunk(Some(chunk))),
            vec![(1, 0, "b".into())],
        );
        let out = op.next_factorized().unwrap().unwrap();
        assert_eq!(out.level_count(), 2, "ancestor spine kept for path count");
        assert_eq!(out.logical_row_count(), 3);
        assert_eq!(out.flatten().row_count(), 3);
    }
}
