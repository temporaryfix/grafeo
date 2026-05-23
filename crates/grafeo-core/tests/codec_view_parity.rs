//! Parity regression test: `*View` borrowing readers produce identical
//! results to the owned codecs across arbitrary inputs.
//!
//! ```bash
//! cargo test -p grafeo-core --test codec_view_parity
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test codec_view_parity
//! ```

use grafeo_common::types::NodeId;
use grafeo_core::codec::{FsstCodec, FsstView, WebGraphBuilder, WebGraphView};
use grafeo_core::index::vector::{RabitqView, TwoStageVectorIndex};
use proptest::prelude::*;

// ── FSST parity ──────────────────────────────────────────────────

fn string_set() -> impl Strategy<Value = Vec<Vec<u8>>> {
    proptest::collection::vec(
        proptest::collection::vec(any::<u8>(), 0..=32),
        0..=16,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// FsstView::get matches FsstCodec::get for every string index.
    #[test]
    fn fsst_view_matches_owned(strings in string_set()) {
        let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
        let owned = FsstCodec::build(&refs);
        let blob = bytes::Bytes::from(owned.to_bytes());
        let view = FsstView::open(blob).expect("open");

        prop_assert_eq!(view.len(), owned.len());
        for i in 0..strings.len() {
            let owned_s = owned.get(i).expect("owned").expect("decode");
            let view_s = view.get(i).expect("view").expect("decode");
            prop_assert_eq!(&view_s, &owned_s);
        }
    }
}

// ── WebGraph parity ──────────────────────────────────────────────

fn webgraph_input() -> impl Strategy<Value = (u64, Vec<(u64, u64)>)> {
    (1u64..=20).prop_flat_map(|n| {
        let edges = proptest::collection::vec((0u64..n, 0u64..n), 0..=60);
        (Just(n), edges)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// WebGraphView::successors matches WebGraphCodec::successors for every node.
    #[test]
    fn webgraph_view_matches_owned((num_nodes, edges) in webgraph_input()) {
        let mut b = WebGraphBuilder::new(num_nodes);
        for &(s, d) in &edges {
            b.add_edge(s, d).unwrap();
        }
        let owned = b.build();
        let blob = bytes::Bytes::from(owned.to_bytes());
        let view = WebGraphView::open(blob).expect("open");

        prop_assert_eq!(view.num_nodes(), owned.num_nodes());
        prop_assert_eq!(view.num_edges(), owned.num_edges());
        for u in 0..owned.num_nodes() {
            let owned_succ: Vec<u64> = owned.successors(u).collect();
            let view_succ: Vec<u64> = view.successors(u).collect();
            prop_assert_eq!(view_succ, owned_succ);
        }
    }
}

// ── RaBitQ parity ────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.f32().max(f32::MIN_POSITIVE);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RabitqView::search matches TwoStageVectorIndex::search.
    #[test]
    fn rabitq_view_matches_owned(
        seed in any::<u64>(),
        count in 5usize..=40,
    ) {
        let dim = 32;
        let mut rng = Rng(seed);
        let vectors: Vec<(NodeId, Vec<f32>)> = (0..count)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
                (NodeId::new(i as u64 + 1), v)
            })
            .collect();
        let owned = TwoStageVectorIndex::build(&vectors, dim, seed ^ 0xAA);
        let blob = bytes::Bytes::from(owned.to_bytes());
        let view = RabitqView::open(blob).expect("open");

        prop_assert_eq!(view.len(), owned.len());

        // reason: index is modulo vectors.len() so truncation on 32-bit is safe
        #[allow(clippy::cast_possible_truncation)]
        let query = vectors[(seed as usize) % vectors.len()].1.clone();
        let k = 5;
        prop_assert_eq!(view.search(&query, k, 8), owned.search(&query, k, 8));
    }
}
