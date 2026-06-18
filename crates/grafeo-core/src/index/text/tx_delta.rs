//! Per-transaction text-index delta for snapshot-versioned BM25.
//!
//! Buffered (uncommitted) text-property writes are recorded here instead of
//! mutating the committed [`InvertedIndex`] directly.  This preserves the
//! committed index as the snapshot-consistent base for all other readers while
//! the writing transaction can see its own edits (read-your-writes — TI4).
//!
//! Lifecycle:
//! - **buffer_set / buffer_remove** — called from the transactional
//!   `set_node_property_buffered` / `remove_node_property_buffered` paths when
//!   the property key is covered by a text index.
//! - **TI4 (search)** — `changes_for` lets the search path merge the delta over
//!   the committed postings for the writing transaction.
//! - **TI5 (promote)** — `changes_for` is consumed at commit to build the
//!   versioned posting entries for the committed index.
//! - **rollback** — the delta is simply dropped.

#![cfg(feature = "text-index")]

use grafeo_common::types::NodeId;
use grafeo_common::utils::hash::FxHashMap;

/// Per-transaction uncommitted text-index writes.
///
/// Keyed by `index_key` (`"label:prop"` string), then by `NodeId`.  The inner
/// value is `Some(text)` for a set and `None` for a removal (tombstone).
///
/// Only the **latest** buffered state for each `(index_key, node)` pair is
/// kept — intermediate overwrites within the same transaction are collapsed
/// automatically by `buffer_set` / `buffer_remove`.
#[derive(Debug, Default, Clone)]
pub struct TextIndexDelta {
    // index_key -> (node -> latest buffered text; None = removed)
    changes: FxHashMap<String, FxHashMap<NodeId, Option<String>>>,
}

impl TextIndexDelta {
    /// Records a text-property set for `node` under `index_key`.
    ///
    /// Overwrites any earlier buffered state for this `(index_key, node)` pair.
    pub fn buffer_set(&mut self, index_key: &str, node: NodeId, text: String) {
        self.changes
            .entry(index_key.to_owned())
            .or_default()
            .insert(node, Some(text));
    }

    /// Records a text-property removal (tombstone) for `node` under `index_key`.
    ///
    /// Overwrites any earlier buffered state for this `(index_key, node)` pair.
    pub fn buffer_remove(&mut self, index_key: &str, node: NodeId) {
        self.changes
            .entry(index_key.to_owned())
            .or_default()
            .insert(node, None);
    }

    /// Returns the latest buffered state for `(index_key, node)`, if any.
    ///
    /// - `None` — no buffered change for this pair.
    /// - `Some(&Some(text))` — buffered set.
    /// - `Some(&None)` — buffered removal (tombstone).
    #[must_use]
    pub fn get(&self, index_key: &str, node: NodeId) -> Option<&Option<String>> {
        self.changes.get(index_key)?.get(&node)
    }

    /// Iterates all buffered changes for `index_key`.
    ///
    /// Yields `(NodeId, &Option<String>)` pairs.  Used by TI4 (search merge)
    /// and TI5 (commit-promote).
    pub fn changes_for(&self, index_key: &str) -> impl Iterator<Item = (NodeId, &Option<String>)> {
        self.changes
            .get(index_key)
            .into_iter()
            .flat_map(|m| m.iter().map(|(&node, opt)| (node, opt)))
    }

    /// Returns `true` if no changes have been buffered at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;

    #[test]
    fn buffer_set_then_get_returns_text() {
        let mut delta = TextIndexDelta::default();
        let node = NodeId::new(1);
        delta.buffer_set("Doc:content", node, "hello world".to_owned());

        let result = delta.get("Doc:content", node);
        assert_eq!(result, Some(&Some("hello world".to_owned())));
    }

    #[test]
    fn buffer_remove_returns_none_sentinel() {
        let mut delta = TextIndexDelta::default();
        let node = NodeId::new(2);
        delta.buffer_remove("Doc:content", node);

        let result = delta.get("Doc:content", node);
        assert_eq!(result, Some(&None));
    }

    #[test]
    fn get_returns_none_when_no_change() {
        let delta = TextIndexDelta::default();
        let node = NodeId::new(3);

        assert_eq!(delta.get("Doc:content", node), None);
    }

    #[test]
    fn later_set_overwrites_earlier_remove() {
        let mut delta = TextIndexDelta::default();
        let node = NodeId::new(4);
        delta.buffer_remove("Doc:content", node);
        delta.buffer_set("Doc:content", node, "overwritten".to_owned());

        assert_eq!(
            delta.get("Doc:content", node),
            Some(&Some("overwritten".to_owned()))
        );
    }

    #[test]
    fn later_remove_overwrites_earlier_set() {
        let mut delta = TextIndexDelta::default();
        let node = NodeId::new(5);
        delta.buffer_set("Doc:content", node, "initial".to_owned());
        delta.buffer_remove("Doc:content", node);

        assert_eq!(delta.get("Doc:content", node), Some(&None));
    }

    #[test]
    fn changes_for_iterates_all_entries() {
        let mut delta = TextIndexDelta::default();
        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);
        let n3 = NodeId::new(3);
        delta.buffer_set("Doc:content", n1, "alpha".to_owned());
        delta.buffer_set("Doc:content", n2, "beta".to_owned());
        delta.buffer_remove("Doc:content", n3);

        let mut pairs: Vec<(NodeId, Option<String>)> = delta
            .changes_for("Doc:content")
            .map(|(node, opt)| (node, opt.clone()))
            .collect();
        pairs.sort_by_key(|(n, _)| n.as_u64());

        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], (n1, Some("alpha".to_owned())));
        assert_eq!(pairs[1], (n2, Some("beta".to_owned())));
        assert_eq!(pairs[2], (n3, None));
    }

    #[test]
    fn changes_for_unknown_key_yields_nothing() {
        let delta = TextIndexDelta::default();
        assert_eq!(delta.changes_for("Unknown:prop").count(), 0);
    }

    #[test]
    fn is_empty_true_when_fresh() {
        let delta = TextIndexDelta::default();
        assert!(delta.is_empty());
    }

    #[test]
    fn is_empty_false_after_buffer_set() {
        let mut delta = TextIndexDelta::default();
        delta.buffer_set("X:y", NodeId::new(1), "text".to_owned());
        assert!(!delta.is_empty());
    }
}
