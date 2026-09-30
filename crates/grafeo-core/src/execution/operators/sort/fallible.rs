//! Allocation-bounded, comparator-fallible stable sorting primitives.

use std::cmp::Ordering;

use crate::execution::operators::OperatorError;

/// Stable sort using two exact index vectors and an in-place permutation.
pub(crate) fn try_stable_sort_by<T, F>(
    values: &mut [T],
    mut compare: F,
) -> Result<(), OperatorError>
where
    F: FnMut(&T, &T) -> Result<Ordering, OperatorError>,
{
    let len = values.len();
    let mut indices = reserve_indices(len, "stable sort indices")?;
    indices.extend(0..len);
    let mut scratch = reserve_indices(len, "stable sort scratch indices")?;
    scratch.resize(len, 0);

    // Bottom-up merge sorting leaves the values untouched until every
    // fallible comparison has succeeded. Already ordered runs avoid all
    // element comparisons except their boundary check.
    let mut width = 1;
    let mut source_is_indices = true;
    while width < len {
        let block = width.checked_mul(2).unwrap_or(len);
        let (source, destination) = if source_is_indices {
            (&indices, &mut scratch)
        } else {
            (&scratch, &mut indices)
        };
        for start in (0..len).step_by(block) {
            let mid = start.saturating_add(width).min(len);
            let end = start.saturating_add(block).min(len);
            if mid == end || comes_before(source[mid - 1], source[mid], values, &mut compare)? {
                destination[start..end].copy_from_slice(&source[start..end]);
                continue;
            }
            let (mut left, mut right) = (start, mid);
            for output in &mut destination[start..end] {
                if left == mid {
                    *output = source[right];
                    right += 1;
                } else if right == end
                    || comes_before(source[left], source[right], values, &mut compare)?
                {
                    *output = source[left];
                    left += 1;
                } else {
                    *output = source[right];
                    right += 1;
                }
            }
        }
        source_is_indices = !source_is_indices;
        if width > len / 2 {
            break;
        }
        width *= 2;
    }
    if !source_is_indices {
        std::mem::swap(&mut indices, &mut scratch);
    }

    // indices[target] is the source position. Resolve each permutation cycle
    // with safe swaps and mark visited entries in the same index allocation.
    for start in 0..len {
        let mut current = start;
        while indices[current] != start {
            let next = indices[current];
            values.swap(current, next);
            indices[current] = current;
            current = next;
        }
        indices[current] = current;
    }
    Ok(())
}

fn reserve_indices(len: usize, container: &'static str) -> Result<Vec<usize>, OperatorError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(len)
        .map_err(|source| OperatorError::ResidentContainerAllocation { container, source })?;
    if indices.capacity() != len {
        return Err(OperatorError::ResidentContainerInvariant {
            container,
            message: "exact index capacity contract violated",
        });
    }
    Ok(indices)
}

fn comes_before<T, F>(
    left: usize,
    right: usize,
    values: &[T],
    compare: &mut F,
) -> Result<bool, OperatorError>
where
    F: FnMut(&T, &T) -> Result<Ordering, OperatorError>,
{
    let ordering = compare(&values[left], &values[right])?;
    Ok(ordering == Ordering::Less || (ordering == Ordering::Equal && left < right))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sort(values: &mut [i32]) -> Result<(), OperatorError> {
        try_stable_sort_by(values, |left, right| Ok(left.cmp(right)))
    }

    #[test]
    fn stable_ascending_and_descending_ties() {
        let mut ascending = [(2, 0), (1, 0), (2, 1), (1, 1)];
        try_stable_sort_by(&mut ascending, |left, right| Ok(left.0.cmp(&right.0))).unwrap();
        assert_eq!(ascending, [(1, 0), (1, 1), (2, 0), (2, 1)]);

        let mut descending = [(1, 0), (2, 0), (1, 1), (2, 1)];
        try_stable_sort_by(&mut descending, |left, right| Ok(right.0.cmp(&left.0))).unwrap();
        assert_eq!(descending, [(2, 0), (2, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn handles_empty_single_and_many() {
        let mut empty: [i32; 0] = [];
        sort(&mut empty).unwrap();
        let mut one = [1];
        sort(&mut one).unwrap();
        let mut many = [4, 1, 3, 2, 0];
        sort(&mut many).unwrap();
        assert_eq!(many, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn shuffled_duplicate_runs_match_stable_reference() {
        let mut values: Vec<_> = (0..120)
            .map(|original| ((original * 37) % 12, original))
            .collect();
        let mut expected = values.clone();
        expected.sort_by_key(|value| value.0);
        try_stable_sort_by(&mut values, |left, right| Ok(left.0.cmp(&right.0))).unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn comparison_error_leaves_values_untouched() {
        let mut values = [4, 1, 3, 2];
        let original = values;
        let error = try_stable_sort_by(&mut values, |left, right| {
            if *left == 3 || *right == 3 {
                Err(OperatorError::Execution(
                    "injected comparison failure".into(),
                ))
            } else {
                Ok(left.cmp(right))
            }
        });
        assert!(error.is_err());
        assert_eq!(values, original);
    }
}
