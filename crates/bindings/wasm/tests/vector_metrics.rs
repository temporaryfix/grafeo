//! Exercise the actual WASM SIMD dispatcher with unaligned inputs and tails.
#![cfg(all(target_arch = "wasm32", feature = "rabitq-codec"))]

use grafeo_core::index::vector::{
    cosine_distance, dot_product, euclidean_distance_squared, manhattan_distance, simd_support,
};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn vector_metrics_handle_unaligned_blocks_and_tails() {
    assert_eq!(simd_support(), "wasm-simd128");
    let left: Vec<f32> = (1_u16..=24).map(f32::from).collect();
    let right: Vec<f32> = (1_u16..=24).rev().map(f32::from).collect();
    let mut unaligned_pairs = 0;
    for left_offset in 0..4 {
        for right_offset in 0..4 {
            for length in [0, 1, 3, 4, 5, 7, 17] {
                let left = &left[left_offset..left_offset + length];
                let right = &right[right_offset..right_offset + length];
                if length >= 4
                    && left.as_ptr().align_offset(16) != 0
                    && right.as_ptr().align_offset(16) != 0
                {
                    unaligned_pairs += 1;
                }
                let expected_dot: f32 = left.iter().zip(right).map(|(x, y)| x * y).sum();
                let expected_squared: f32 =
                    left.iter().zip(right).map(|(x, y)| (x - y).powi(2)).sum();
                let expected_manhattan: f32 =
                    left.iter().zip(right).map(|(x, y)| (x - y).abs()).sum();
                let left_norm: f32 = left.iter().map(|value| value * value).sum();
                let right_norm: f32 = right.iter().map(|value| value * value).sum();
                let expected_cosine =
                    1.0 - expected_dot / (left_norm.sqrt() * right_norm.sqrt() + f32::EPSILON);
                for (actual, expected) in [
                    (dot_product(left, right), expected_dot),
                    (euclidean_distance_squared(left, right), expected_squared),
                    (manhattan_distance(left, right), expected_manhattan),
                    (cosine_distance(left, right), expected_cosine),
                ] {
                    assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
                }
            }
        }
    }
    assert_eq!(unaligned_pairs, 36);
}
