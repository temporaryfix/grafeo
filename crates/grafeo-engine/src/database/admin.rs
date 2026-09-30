//! Admin, introspection, and diagnostic operations for GrafeoDB.

#[cfg(feature = "wal")]
use std::path::Path;

use grafeo_common::utils::error::Result;

impl super::GrafeoDB {
    // =========================================================================
    // ADMIN API: Counts
    // =========================================================================

    /// Returns the number of nodes in the database.
    #[must_use]
    pub fn node_count(&self) -> usize {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().node_count()
    }

    /// Returns the number of edges in the database.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().edge_count()
    }

    /// Returns the number of distinct labels in the database.
    #[must_use]
    pub fn label_count(&self) -> usize {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().all_labels().len()
    }

    /// Returns the number of distinct property keys in the database.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().all_property_keys().len()
    }

    /// Returns the number of distinct edge types in the database.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        let _publication = self.transaction_manager.publication().read();
        self.read_graph_view().all_edge_types().len()
    }

    // =========================================================================
    // ADMIN API: Introspection
    // =========================================================================

    /// Returns a hierarchical memory usage breakdown.
    ///
    /// Walks all internal structures (store, indexes, MVCC chains, caches,
    /// string pools, buffer manager) and returns estimated heap bytes for each.
    /// Safe to call concurrently with queries.
    #[must_use]
    pub fn memory_usage(&self) -> crate::memory_usage::MemoryUsage {
        use crate::memory_usage::{BufferManagerMemory, CacheMemory, MemoryUsage};
        use grafeo_common::memory::MemoryRegion;

        let _publication = self.transaction_manager.publication().read();
        let (store, indexes, mvcc, string_pool) = self.lpg_store().memory_breakdown();

        let (parsed_bytes, optimized_bytes, cached_plan_count) =
            self.query_cache.heap_memory_bytes();
        let mut caches = CacheMemory {
            parsed_plan_cache_bytes: parsed_bytes,
            optimized_plan_cache_bytes: optimized_bytes,
            cached_plan_count,
            ..Default::default()
        };
        caches.compute_total();

        let bm_stats = self.buffer_manager.stats();
        let buffer_manager = BufferManagerMemory {
            budget_bytes: bm_stats.budget,
            allocated_bytes: bm_stats.total_allocated,
            graph_storage_bytes: bm_stats.region_usage(MemoryRegion::GraphStorage),
            index_buffers_bytes: bm_stats.region_usage(MemoryRegion::IndexBuffers),
            execution_buffers_bytes: bm_stats.region_usage(MemoryRegion::ExecutionBuffers),
            spill_staging_bytes: bm_stats.region_usage(MemoryRegion::SpillStaging),
        };

        let mut usage = MemoryUsage {
            store,
            indexes,
            mvcc,
            caches,
            string_pool,
            buffer_manager,
            ..Default::default()
        };

        #[cfg(feature = "triple-store")]
        {
            use crate::memory_usage::RdfMemory;
            let (
                triple_count,
                triples_and_indexes_bytes,
                term_dictionary_bytes,
                ring_index_bytes,
                named_graph_count,
            ) = self.rdf_store.heap_memory_bytes();
            usage.rdf = RdfMemory {
                triple_count,
                triples_and_indexes_bytes,
                term_dictionary_bytes,
                ring_index_bytes,
                named_graph_count,
                total_bytes: 0,
            };
            usage.rdf.compute_total();
        }

        #[cfg(feature = "cdc")]
        {
            use crate::memory_usage::CdcMemory;
            let (total_bytes, entity_count, event_count) = self.cdc_log.heap_memory_bytes();
            usage.cdc = CdcMemory {
                total_bytes,
                entity_count,
                event_count,
            };
        }

        usage.compute_total();
        usage
    }

    /// Returns detailed database statistics.
    ///
    /// Includes counts, memory usage, and index information.
    #[must_use]
    pub fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        #[cfg(feature = "wal")]
        let disk_bytes = self.config.path.as_ref().and_then(|p| {
            if p.exists() {
                Self::calculate_disk_usage(p).ok()
            } else {
                None
            }
        });
        #[cfg(not(feature = "wal"))]
        let disk_bytes: Option<usize> = None;

        crate::admin::DatabaseStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            label_count: self.label_count(),
            edge_type_count: self.edge_type_count(),
            property_key_count: self.property_key_count(),
            index_count: self.catalog.index_count(),
            memory_bytes: self.memory_usage().total_bytes,
            disk_bytes,
        }
    }

    /// Calculates total disk usage for the database directory.
    #[cfg(feature = "wal")]
    fn calculate_disk_usage(path: &Path) -> Result<usize> {
        let mut total = 0usize;
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    // reason: file sizes fit usize on 64-bit targets
                    #[allow(clippy::cast_possible_truncation)]
                    let file_len = metadata.len() as usize;
                    total += file_len;
                } else if metadata.is_dir() {
                    total += Self::calculate_disk_usage(&entry.path())?;
                }
            }
        }
        Ok(total)
    }

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// For LPG mode, returns label and edge type information.
    /// For RDF mode, returns predicate and named graph information.
    #[must_use]
    pub fn schema(&self) -> crate::admin::SchemaInfo {
        let _publication = self.transaction_manager.publication().read();
        let view = self.read_graph_view();
        let labels = view
            .all_labels()
            .into_iter()
            .map(|name| crate::admin::LabelInfo {
                count: view.nodes_by_label(&name).len(),
                name,
            })
            .collect();

        // Per-edge-type counts, tier-merged: tally the merged edge set once
        // (the raw overlay is empty after compact()).
        let mut edge_type_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for edge in self.read_all_edges() {
            *edge_type_counts
                .entry(edge.edge_type.to_string())
                .or_default() += 1;
        }
        let edge_types = view
            .all_edge_types()
            .into_iter()
            .map(|name| crate::admin::EdgeTypeInfo {
                count: edge_type_counts.get(&name).copied().unwrap_or(0),
                name,
            })
            .collect();

        let property_keys = view.all_property_keys();

        crate::admin::SchemaInfo::Lpg(crate::admin::LpgSchemaInfo {
            labels,
            edge_types,
            property_keys,
        })
    }

    /// Returns detailed information about all indexes, using their canonical
    /// explicit or catalog-allocated owner names.
    #[must_use]
    pub fn list_indexes(&self) -> Vec<crate::admin::IndexInfo> {
        let _publication = self.transaction_manager.publication().read();
        let catalog = self.catalog.read();
        catalog
            .all_indexes()
            .into_iter()
            .map(|def| {
                let label_name = catalog
                    .get_label_name(def.label)
                    .unwrap_or_else(|| "?".into());
                let prop_name = catalog
                    .get_property_key_name(def.property_key)
                    .unwrap_or_else(|| "?".into());
                crate::admin::IndexInfo {
                    name: def.name,
                    index_type: format!("{:?}", def.index_type),
                    target: format!("{}:{}", label_name, prop_name),
                    unique: false,
                    cardinality: None,
                    size_bytes: None,
                }
            })
            .collect()
    }

    /// Validates database integrity.
    ///
    /// Checks for:
    /// - Dangling edge references (edges pointing to non-existent nodes)
    /// - Internal index consistency
    ///
    /// Returns a list of errors and warnings. Empty errors = valid.
    #[must_use]
    pub fn validate(&self) -> crate::admin::ValidationResult {
        let _publication = self.transaction_manager.publication().read();
        let mut result = crate::admin::ValidationResult::default();

        // Check for dangling edge references (tier-merged).
        let view = self.read_graph_view();
        for edge in self.read_all_edges() {
            if view.get_node(edge.src).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_SRC".to_string(),
                    message: format!(
                        "Edge {} references non-existent source node {}",
                        edge.id.0, edge.src.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
            if view.get_node(edge.dst).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_DST".to_string(),
                    message: format!(
                        "Edge {} references non-existent destination node {}",
                        edge.id.0, edge.dst.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
        }

        // Add warnings for potential issues
        if view.node_count() > 0 && view.edge_count() == 0 {
            result.warnings.push(crate::admin::ValidationWarning {
                code: "NO_EDGES".to_string(),
                message: "Database has nodes but no edges".to_string(),
                context: None,
            });
        }

        result
    }

    /// Returns WAL (Write-Ahead Log) status.
    ///
    /// Returns a disabled status if WAL is not enabled.
    ///
    /// # Errors
    /// Returns the underlying WAL telemetry error.
    pub fn wal_status(&self) -> Result<crate::admin::WalStatus> {
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            return Ok(crate::admin::WalStatus {
                enabled: true,
                path: self.config.path.as_ref().map(|p| p.join("wal")),
                size_bytes: wal.size_bytes()?,
                // reason: WAL record count fits usize on 64-bit targets
                #[allow(clippy::cast_possible_truncation)]
                record_count: wal.record_count() as usize,
                last_checkpoint: wal.last_checkpoint_timestamp()?,
                current_epoch: self.lpg_store().current_epoch().as_u64(),
            });
        }

        Ok(crate::admin::WalStatus {
            enabled: false,
            path: None,
            size_bytes: 0,
            record_count: 0,
            last_checkpoint: None,
            current_epoch: self.lpg_store().current_epoch().as_u64(),
        })
    }
}

#[cfg(all(test, feature = "lpg"))]
mod index_info {
    use super::super::GrafeoDB;
    use crate::{CreateIndexRequest, IndexCreateKind};
    use grafeo_common::types::GraphPath;

    #[test]
    fn list_indexes_reports_actual_explicit_and_anonymous_owners()
    -> Result<(), Box<dyn std::error::Error>> {
        let db = GrafeoDB::new_in_memory();
        let anonymous = db.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: None,
            label: None,
            property: "body".into(),
            kind: IndexCreateKind::Property,
        })?;
        let explicit = db.create_index(CreateIndexRequest {
            graph: GraphPath::root(),
            name: Some("declared-owner".into()),
            label: None,
            property: "title".into(),
            kind: IndexCreateKind::BTree,
        })?;
        let anonymous_name = format!("@grafeo-index:{}", anonymous.as_u32());
        let mut names: Vec<_> = db
            .list_indexes()
            .into_iter()
            .map(|info| info.name)
            .collect();
        names.sort();
        assert_eq!(names, [anonymous_name.clone(), "declared-owner".into()]);
        assert!(db.drop_index(explicit)?);
        let remaining = db.list_indexes();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].name, anonymous_name);
        assert!(db.drop_index(anonymous)?);
        assert!(db.list_indexes().is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod publication_checkpoint {
    use super::super::GrafeoDB;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    /// `wal_checkpoint` takes the publication write lock, so it cannot retire
    /// WAL while a commit (or another checkpoint) holds that lock.
    #[test]
    fn wal_checkpoint_waits_for_publication_write() {
        let db = Arc::new(GrafeoDB::new_in_memory());
        let held = db.transaction_manager.publication().write();
        let db_ckpt = Arc::clone(&db);
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            started_tx.send(()).ok();
            let _ = db_ckpt.wal_checkpoint();
            done_tx.send(()).ok();
        });
        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("checkpoint thread started");
        std::thread::sleep(Duration::from_millis(80));
        assert!(
            done_rx.try_recv().is_err(),
            "wal_checkpoint must block while publication write is held"
        );
        drop(held);
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("wal_checkpoint completes after publication write is released");
        handle.join().expect("checkpoint thread");
    }
}
