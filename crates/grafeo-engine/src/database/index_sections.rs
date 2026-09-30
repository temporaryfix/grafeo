//! One coherent LPG graph cut for catalog and exact derived-index sections.
//!
//! Container builders hold transaction publication while capturing this
//! owned topology.  Reusing the same `Arc` set for Catalog, VectorStore, and
//! TextIndex prevents independently enumerated named-graph generations from
//! describing different worlds.

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::collections::BTreeSet;
use std::sync::Arc;

use grafeo_common::types::GraphPath;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::utils::error::Error;
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection};
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_core::graph::lpg::{PhysicalIndexKey, decode_index_key};

/// Owned recursive topology. Callers retain engine publication through encoding.
pub(super) struct LpgIndexGraphCut {
    graphs: Vec<(GraphPath, Arc<LpgStore>)>,
}

impl LpgIndexGraphCut {
    pub(super) fn capture(root: Arc<LpgStore>) -> Result<Self> {
        Ok(Self::from_graphs(
            LpgStoreSection::new(root).capture_graphs()?,
        ))
    }

    pub(super) fn from_graphs(graphs: Vec<(GraphPath, Arc<LpgStore>)>) -> Self {
        Self { graphs }
    }

    pub(super) fn graphs(&self) -> Vec<(GraphPath, Arc<LpgStore>)> {
        self.graphs.clone()
    }

    #[cfg(feature = "vector-index")]
    pub(super) fn vector_views(
        &self,
    ) -> Result<
        Vec<(
            PhysicalIndexKey,
            grafeo_core::index::vector::VectorIndexView,
        )>,
    > {
        let mut views = Vec::new();
        for (path, graph) in &self.graphs {
            self.extend_vector_views(path, graph, &mut views)?;
        }
        views.sort_by(|left, right| left.0.cmp(&right.0));
        reject_duplicate_keys(&views, "vector")?;
        Ok(views)
    }

    #[cfg(feature = "vector-index")]
    fn extend_vector_views(
        &self,
        graph: &GraphPath,
        store: &LpgStore,
        views: &mut Vec<(
            PhysicalIndexKey,
            grafeo_core::index::vector::VectorIndexView,
        )>,
    ) -> Result<()> {
        for (key, view) in store.vector_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or_else(|| {
                Error::Serialization(format!(
                    "{graph:?} graph vector registry contains invalid index key {key:?}"
                ))
            })?;
            views.push((
                PhysicalIndexKey::vector(graph.clone(), label, property),
                view,
            ));
        }
        Ok(())
    }

    #[cfg(feature = "text-index")]
    pub(super) fn text_views(
        &self,
    ) -> Result<Vec<(PhysicalIndexKey, grafeo_core::index::text::TextIndexView)>> {
        let mut views = Vec::new();
        for (path, graph) in &self.graphs {
            self.extend_text_views(path, graph, &mut views)?;
        }
        views.sort_by(|left, right| left.0.cmp(&right.0));
        reject_duplicate_keys(&views, "text")?;
        Ok(views)
    }

    #[cfg(feature = "text-index")]
    fn extend_text_views(
        &self,
        graph: &GraphPath,
        store: &LpgStore,
        views: &mut Vec<(PhysicalIndexKey, grafeo_core::index::text::TextIndexView)>,
    ) -> Result<()> {
        for (key, view) in store.text_index_entries() {
            let (label, property) = decode_index_key(&key).ok_or_else(|| {
                Error::Serialization(format!(
                    "{graph:?} graph text registry contains invalid index key {key:?}"
                ))
            })?;
            views.push((PhysicalIndexKey::text(graph.clone(), label, property), view));
        }
        Ok(())
    }
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn reject_duplicate_keys<T>(entries: &[(PhysicalIndexKey, T)], kind: &str) -> Result<()> {
    let mut keys = BTreeSet::new();
    for (key, _) in entries {
        if !keys.insert(key) {
            return Err(Error::Serialization(format!(
                "duplicate canonical scoped {kind} index key {key:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_cut_preserves_recursive_literal_paths_and_incarnations() -> Result<()> {
        let root = Arc::new(LpgStore::new()?);
        let empty = root.graph_or_create("")?;
        let literal = root.graph_or_create("outer/inner")?;
        let outer = root.graph_or_create("outer")?;
        let nested = outer.graph_or_create("inner")?;
        let named_default = root.graph_or_create("default")?;
        let cut = LpgIndexGraphCut::capture(Arc::clone(&root))?;
        let expected = [
            (vec![], root),
            (vec![""], empty),
            (vec!["default"], named_default),
            (vec!["outer"], outer),
            (vec!["outer", "inner"], nested),
            (vec!["outer/inner"], literal),
        ];
        assert_eq!(cut.graphs.len(), expected.len());
        for ((path, store), (components, expected_store)) in cut.graphs.iter().zip(expected) {
            assert_eq!(path.components(), components);
            assert!(Arc::ptr_eq(store, &expected_store));
        }
        Ok(())
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn vector_views_encode_every_graph_scope_canonically() {
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

        let root = Arc::new(LpgStore::new().expect("root store"));
        let named = root.graph_or_create("").expect("empty-name graph");
        root.add_vector_index(
            "Doc",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Cosine,
            )))),
        );
        named.add_vector_index(
            "Doc",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Cosine,
            )))),
        );

        let cut = LpgIndexGraphCut::capture(root).expect("capture graph cut");
        let keys: Vec<_> = cut
            .vector_views()
            .expect("collect vector views")
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(
            keys,
            vec![
                PhysicalIndexKey::vector(GraphPath::root(), "Doc", "embedding"),
                PhysicalIndexKey::vector(
                    GraphPath::root().child("").expect("path"),
                    "Doc",
                    "embedding"
                ),
            ]
        );
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn text_views_encode_every_graph_scope_canonically() {
        use grafeo_core::index::text::{BM25Config, InvertedIndex};
        use parking_lot::RwLock;

        let root = Arc::new(LpgStore::new().expect("root store"));
        let named = root
            .graph_or_create("default")
            .expect("named default graph");
        root.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
        named.add_text_index(
            "Doc",
            "body",
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );

        let cut = LpgIndexGraphCut::capture(root).expect("capture graph cut");
        let keys: Vec<_> = cut
            .text_views()
            .expect("collect text views")
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(
            keys,
            vec![
                PhysicalIndexKey::text(GraphPath::root(), "Doc", "body"),
                PhysicalIndexKey::text(
                    GraphPath::root().child("default").expect("path"),
                    "Doc",
                    "body"
                ),
            ]
        );
    }
}
