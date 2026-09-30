//! Graph-qualified index owners. Creation commits one logical/physical owner;
//! removal and rebuilding address that exact owner, never a mutable name.

use grafeo_common::types::{GraphPath, IndexId, NodeId, Value};
use grafeo_common::utils::error::Result;

/// A graph-qualified index creation request.
#[derive(Clone, Debug)]
pub struct CreateIndexRequest {
    /// Component-qualified graph path. An empty path selects the root graph.
    pub graph: GraphPath,
    /// Optional user name. Anonymous indexes receive an engine-owned name.
    pub name: Option<String>,
    /// Required for Text/Vector; absent for property-wide Property/BTree.
    pub label: Option<String>,
    /// Property to index.
    pub property: String,
    /// Index family and requested configuration, resolved before commit.
    pub kind: IndexCreateKind,
}

/// Requested index family. Disabled families return an error, not a no-op.
#[derive(Clone, Debug)]
pub enum IndexCreateKind {
    /// Property-wide hash lookup.
    Property,
    /// Property-wide ordered lookup declaration.
    BTree,
    /// BM25 text search with the built-in Simple tokenizer.
    Text {
        /// Minimum lowercased UTF-8 token byte length; defaults to 2. Zero is valid.
        min_token_length: Option<usize>,
    },
    /// Approximate nearest-neighbor search.
    Vector {
        /// Vector width; inferred from existing values when absent.
        dimensions: Option<usize>,
        /// Distance metric; defaults to cosine.
        metric: Option<String>,
        /// HNSW links per node.
        m: Option<usize>,
        /// HNSW construction beam width.
        ef_construction: Option<usize>,
        /// HNSW query beam width; defaults to the existing index search depth.
        ef: Option<usize>,
        /// Optional vector quantization mode.
        quantization: Option<String>,
    },
}

impl super::GrafeoDB {
    /// Creates an index and returns its committed, non-reusable owner ID.
    ///
    /// # Errors
    /// Rejects invalid requests, duplicate names/physical targets, unavailable
    /// features, changed graph ownership, or failed publication/durability.
    pub fn create_index(&self, request: CreateIndexRequest) -> Result<IndexId> {
        self.session().create_index_durable(request)
    }

    /// Drops exactly this owner. An absent owner returns `false`.
    ///
    /// # Errors
    /// Returns publication/durability and concurrent owner-change errors.
    pub fn drop_index(&self, owner: IndexId) -> Result<bool> {
        self.session().drop_index_durable(owner)
    }

    /// Replaces an index's contents in one commit, retaining its owner and full
    /// resolved configuration. Failure leaves the original index in place.
    ///
    /// # Errors
    /// Rejects an absent/changed owner or any preparation/publication failure.
    pub fn rebuild_index(&self, owner: IndexId) -> Result<()> {
        self.session().rebuild_index_durable(owner)
    }

    /// Returns whether the root graph has a property-wide index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        let _publication = self.transaction_manager.publication().read();
        self.lpg_store().has_property_index(property)
    }

    /// Finds root-graph nodes by property value, using an index when available.
    #[must_use]
    pub fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view()
            .find_nodes_by_property(property, value)
    }
}
