//! End-to-end: a high-cardinality, compressible string column is FSST-encoded
//! by `compact()` (the builder's `encode_string_column` selects FSST over the
//! dictionary for this shape) and reads return the exact original strings.
//!
//! The unit tests in `builder.rs` prove the codec *selection*; this proves the
//! full compact + read path round-trips through whichever codec was chosen.

#![cfg(all(feature = "compact-store", feature = "lpg"))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

fn url(i: usize) -> String {
    format!("https://example.com/users/profile/{i:08}/settings")
}

#[test]
fn fsst_string_column_round_trips_through_compact() {
    let mut db = GrafeoDB::new_in_memory();
    let mut ids = Vec::new();
    for i in 0..500 {
        let n = db.create_node(&["Doc"]);
        db.set_node_property(n, "url", Value::String(url(i).into()))
            .expect("set node property");
        ids.push((n, i));
    }

    db.compact().expect("compact");

    let url_key = PropertyKey::new("url");
    for (n, i) in ids {
        let got = db
            .get_node(n)
            .and_then(|node| node.properties.get(&url_key).cloned());
        assert_eq!(
            got,
            Some(Value::String(url(i).into())),
            "node {i} url must survive FSST encoding through compact()",
        );
    }
}
