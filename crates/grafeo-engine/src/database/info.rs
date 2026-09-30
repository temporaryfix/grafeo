//! Database identity and inspection shared by native model profiles.

use std::path::Path;

impl super::GrafeoDB {
    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return false.
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.config.path.is_some()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.config.path.as_deref()
    }

    /// Returns high-level database information.
    ///
    /// Includes node/edge counts, persistence status, and mode (LPG/RDF).
    #[must_use]
    pub fn info(&self) -> crate::admin::DatabaseInfo {
        let _publication = self.transaction_manager.publication().read();
        #[cfg(feature = "triple-store")]
        let rdf_counts = || {
            let mut subjects = std::collections::HashSet::new();
            subjects.extend(self.rdf_store.subjects());
            let mut triples = self.rdf_store.len();
            for name in self.rdf_store.graph_names() {
                if let Some(graph) = self.rdf_store.graph(&name) {
                    subjects.extend(graph.subjects());
                    triples += graph.len();
                }
            }
            (subjects.len(), triples)
        };
        #[cfg(feature = "lpg")]
        let lpg_counts = |mode| {
            (
                mode,
                self.read_graph_view().node_count(),
                self.read_graph_view().edge_count(),
            )
        };
        #[cfg(not(feature = "lpg"))]
        let lpg_counts = |mode| (mode, 0, 0);
        let (mode, node_count, edge_count) = match self.config.graph_model {
            crate::config::GraphModel::Lpg => lpg_counts(crate::admin::DatabaseMode::Lpg),
            #[cfg(feature = "triple-store")]
            crate::config::GraphModel::Rdf => {
                let (subjects, triples) = rdf_counts();
                (crate::admin::DatabaseMode::Rdf, subjects, triples)
            }
            #[cfg(not(feature = "triple-store"))]
            crate::config::GraphModel::Rdf => (crate::admin::DatabaseMode::Rdf, 0, 0),
            crate::config::GraphModel::Both => lpg_counts(crate::admin::DatabaseMode::Both),
        };
        crate::admin::DatabaseInfo {
            mode,
            node_count,
            edge_count,
            is_persistent: self.is_persistent(),
            path: self.config.path.clone(),
            wal_enabled: self.config.wal_enabled,
            version: env!("CARGO_PKG_VERSION").to_string(),
            features: {
                let mut f = Vec::new();
                if cfg!(feature = "lpg") {
                    f.push("lpg".into());
                }
                if cfg!(feature = "gql") {
                    f.push("gql".into());
                }
                if cfg!(feature = "cypher") {
                    f.push("cypher".into());
                }
                if cfg!(feature = "sparql") {
                    f.push("sparql".into());
                }
                if cfg!(feature = "gremlin") {
                    f.push("gremlin".into());
                }
                if cfg!(feature = "graphql") {
                    f.push("graphql".into());
                }
                if cfg!(feature = "sql-pgq") {
                    f.push("sql-pgq".into());
                }
                if cfg!(feature = "triple-store") {
                    f.push("rdf".into());
                }
                if cfg!(feature = "algos") {
                    f.push("algos".into());
                }
                if cfg!(feature = "vector-index") {
                    f.push("vector-index".into());
                }
                if cfg!(feature = "text-index") {
                    f.push("text-index".into());
                }
                if cfg!(feature = "hybrid-search") {
                    f.push("hybrid-search".into());
                }
                if cfg!(feature = "cdc") {
                    f.push("cdc".into());
                }
                f
            },
        }
    }
}
