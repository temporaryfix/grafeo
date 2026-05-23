//! Round-trip equivalence regression test for the FSST string codec.
//!
//! Mirrors the fork's property-test discipline (see
//! `grafeo-engine/tests/compact_roundtrip_proptest.rs`): a `proptest!`
//! block generating arbitrary string sets, plus fixed-seed regression
//! cases. The invariants:
//!
//! 1. Every input string decodes bit-identical to its original.
//! 2. Random access (`codec.get(i)`) equals sequential decode.
//! 3. The blob round-trips: `from_bytes(codec.to_bytes())` yields a codec
//!    with the same `get` results.
//!
//! ```bash
//! cargo test -p grafeo-core --test fsst_round_trip
//! PROPTEST_CASES=512 cargo test -p grafeo-core --test fsst_round_trip
//! ```

use grafeo_core::codec::FsstCodec;
use proptest::prelude::*;

fn string_strategy() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..=64)
}

fn string_set_strategy() -> impl Strategy<Value = Vec<Vec<u8>>> {
    proptest::collection::vec(string_strategy(), 0..=24)
}

fn check_round_trip(strings: &[Vec<u8>]) {
    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let codec = FsstCodec::build(&refs);

    // 1. Every string decodes bit-identical to its original.
    assert_eq!(codec.len(), strings.len());
    for (i, s) in strings.iter().enumerate() {
        let decoded = codec.get(i).expect("get").expect("decode");
        assert_eq!(&decoded, s, "string {i} decoded mismatched");
    }

    // 2. Blob round-trip preserves all results.
    let blob = codec.to_bytes();
    let reopened = FsstCodec::from_bytes(&blob).expect("from_bytes");
    assert_eq!(reopened.len(), strings.len());
    for (i, s) in strings.iter().enumerate() {
        let decoded = reopened.get(i).expect("get").expect("decode");
        assert_eq!(&decoded, s, "blob-reopened string {i} mismatched");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every input string decodes bit-identical, before and after a blob
    /// round-trip, for arbitrary byte strings.
    #[test]
    fn fsst_round_trip_arbitrary_strings(strings in string_set_strategy()) {
        check_round_trip(&strings);
    }
}

// ── Fixed regression seeds ───────────────────────────────────────

#[test]
fn fsst_round_trip_empty_set() {
    check_round_trip(&[]);
}

#[test]
fn fsst_round_trip_all_empty_strings() {
    check_round_trip(&[vec![], vec![], vec![]]);
}

#[test]
fn fsst_round_trip_all_bytes_0_to_255() {
    let s: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
    check_round_trip(&[s]);
}

#[test]
fn fsst_round_trip_realistic_names() {
    let names = [
        "Vincent Vega",
        "Mia Wallace",
        "Butch Coolidge",
        "Jules Winnfield",
        "Marsellus Wallace",
        "Vincent Vega",  // duplicate
        "Honey Bunny",
        "Pumpkin",
    ];
    let strings: Vec<Vec<u8>> = names.iter().map(|s| s.as_bytes().to_vec()).collect();
    check_round_trip(&strings);
}

#[test]
fn fsst_round_trip_long_string_with_repeats() {
    let mut s = Vec::new();
    for _ in 0..64 {
        s.extend_from_slice(b"the ");
    }
    s.extend_from_slice(b"END");
    check_round_trip(&[s]);
}
