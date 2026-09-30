//! WAL record types and the [`WalEntry`] trait.

use grafeo_common::types::{
    Digest256, EdgeId, EpochId, GraphIncarnationId, GraphPath, NodeId, StoreId, TransactionId,
    Value, WorldIdentityMetadataV1,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

mod graph_paths;

/// Stable decoder diagnostic used to distinguish a structurally forbidden
/// nested catalog batch from unrelated malformed bincode.
pub(crate) const NESTED_CATALOG_BATCH_DECODE_ERROR: &str =
    "CatalogBatchV2 cannot contain a nested catalog batch";

std::thread_local! {
    /// Deserializing a batch's child vector sets this guard before any child
    /// can deserialize its own vector. That bounds the recursive wire shape at
    /// the first nested batch before recursive materialization.
    static CATALOG_BATCH_DECODE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct CatalogBatchDecodeGuard;

impl Drop for CatalogBatchDecodeGuard {
    fn drop(&mut self) {
        CATALOG_BATCH_DECODE_ACTIVE.with(|active| active.set(false));
    }
}

fn deserialize_catalog_batch_records<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<WalRecord>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    CATALOG_BATCH_DECODE_ACTIVE.with(|active| {
        if active.replace(true) {
            return Err(serde::de::Error::custom(NESTED_CATALOG_BATCH_DECODE_ERROR));
        }
        let _guard = CatalogBatchDecodeGuard;
        Vec::<WalRecord>::deserialize(deserializer)
    })
}

/// Trait for WAL record types, enabling type-safe WAL instances.
///
/// [`WalRecord`] is Grafeo's shared LPG/RDF record family. Custom
/// [`TypedWal`](super::TypedWal) users may implement this trait for their own
/// record enum; the wrapper handles durability decisions and transaction
/// semantics without knowing the concrete record type.
pub trait WalEntry: Serialize + DeserializeOwned + Send + Sync + std::fmt::Debug + Clone {
    /// Validates record-local invariants before recovery changes any grouping
    /// or high-water state.
    ///
    /// The default preserves custom WAL compatibility. Implementations should
    /// be deterministic and side-effect free; a returned message is wrapped in
    /// a structured invalid-WAL-entry error with file/offset context.
    ///
    /// # Errors
    ///
    /// Returns a stable explanation when the decoded record cannot safely
    /// participate in recovery.
    fn validate_recovery(&self) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Validates the semantic manifest against the pending physical group.
    /// The iterator includes only this transaction and untagged members.
    ///
    /// # Errors
    /// Returns an explanation for missing or inconsistent group members.
    fn validate_commit_group<'a>(
        &self,
        _members: impl Iterator<Item = &'a Self>,
    ) -> std::result::Result<(), String>
    where
        Self: 'a,
    {
        Ok(())
    }

    /// Final epoch carried by a native commit marker.
    fn commit_epoch(&self) -> Option<EpochId> {
        None
    }

    /// Epoch of a prepared feed batch or its manifest; it must follow prior commits.
    fn feed_epoch(&self) -> Option<EpochId> {
        None
    }

    /// Whether this record should force an immediate fsync in Sync durability mode.
    ///
    /// Returns `true` for commit markers.
    fn requires_sync(&self) -> bool;

    /// Whether this is a transaction commit record.
    fn is_commit(&self) -> bool;

    /// Whether this is a transaction abort record.
    fn is_abort(&self) -> bool;

    /// Whether this is a checkpoint record.
    fn is_checkpoint(&self) -> bool;

    /// Whether this is one complete standalone catalog publication.
    fn is_catalog_batch(&self) -> bool {
        false
    }

    /// Tagged LPG data mutation (not RDF, not commit/abort).
    fn is_lpg_mutation(&self) -> bool {
        false
    }

    /// Whether this record mutates durable graph data (LPG or RDF).
    ///
    /// Failure injection and fail-closed WAL tests use this model-neutral
    /// classification. The default preserves compatibility for custom WAL
    /// record types that only implemented [`Self::is_lpg_mutation`].
    fn is_data_mutation(&self) -> bool {
        self.is_lpg_mutation()
    }

    /// Whether this is a metadata record (e.g., epoch advance).
    ///
    /// Metadata records are not part of any transaction and are always
    /// included in recovery output. They carry structural information
    /// used by backup and point-in-time recovery.
    fn is_metadata(&self) -> bool {
        false
    }

    /// Transaction that owns this record, if tagged.
    ///
    /// Standalone metadata and untagged custom entries return `None`.
    fn transaction_id(&self) -> Option<TransactionId> {
        None
    }

    /// Name carried by a transaction-local savepoint marker.
    ///
    /// Recovery retains these markers only while the owning transaction is
    /// pending. They are consumed before committed records are returned.
    fn savepoint_name(&self) -> Option<&str> {
        None
    }

    /// Name carried by a transaction-local rollback-to-savepoint marker.
    ///
    /// Recovery uses this to discard only the owning transaction's records
    /// after the matching savepoint, leaving interleaved transactions intact.
    fn rollback_to_savepoint_name(&self) -> Option<&str> {
        None
    }

    /// Creates a checkpoint record for this WAL type.
    fn make_checkpoint(transaction_id: TransactionId) -> Self;
}

/// A record in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    // === Schema DDL Records ===
    /// Register a node type definition.
    CreateNodeType {
        /// Type name (corresponds to a label).
        name: String,
        /// Property definitions: (name, data_type, nullable).
        properties: Vec<(String, String, bool)>,
        /// Constraints: (kind, property_names). kind = "unique", "primary_key", "not_null".
        constraints: Vec<(String, Vec<String>)>,
    },

    /// Drop a node type definition.
    DropNodeType {
        /// Type name.
        name: String,
    },

    /// Register an edge type definition.
    CreateEdgeType {
        /// Type name.
        name: String,
        /// Property definitions: (name, data_type, nullable).
        properties: Vec<(String, String, bool)>,
        /// Constraints: (kind, property_names).
        constraints: Vec<(String, Vec<String>)>,
    },

    /// Drop an edge type definition.
    DropEdgeType {
        /// Type name.
        name: String,
    },

    /// Create a constraint.
    CreateConstraint {
        /// Constraint name.
        name: String,
        /// Target label.
        label: String,
        /// Target properties.
        properties: Vec<String>,
        /// Constraint kind: "unique", "node_key", "not_null", "exists".
        kind: String,
    },

    /// Drop a constraint.
    DropConstraint {
        /// Constraint name.
        name: String,
    },

    /// Register a graph type definition.
    CreateGraphType {
        /// Type name.
        name: String,
        /// Allowed node types.
        node_types: Vec<String>,
        /// Allowed edge types.
        edge_types: Vec<String>,
        /// Whether unlisted types are allowed.
        open: bool,
    },

    /// Drop a graph type definition.
    DropGraphType {
        /// Type name.
        name: String,
    },

    /// Register a schema namespace.
    CreateSchema {
        /// Schema name.
        name: String,
    },

    /// Drop a schema namespace.
    DropSchema {
        /// Schema name.
        name: String,
    },

    /// Alter a node type (add/drop properties).
    AlterNodeType {
        /// Type name.
        name: String,
        /// Alterations: ("add", prop_name, type, nullable) or ("drop", prop_name, "", false).
        alterations: Vec<(String, String, String, bool)>,
    },

    /// Alter an edge type (add/drop properties).
    AlterEdgeType {
        /// Type name.
        name: String,
        /// Alterations: ("add", prop_name, type, nullable) or ("drop", prop_name, "", false).
        alterations: Vec<(String, String, String, bool)>,
    },

    /// Alter a graph type (add/drop node/edge types).
    AlterGraphType {
        /// Graph type name.
        name: String,
        /// Alterations: ("add_node_type"|"drop_node_type"|"add_edge_type"|"drop_edge_type", type_name).
        alterations: Vec<(String, String)>,
    },

    /// Create a stored procedure.
    CreateProcedure {
        /// Procedure name.
        name: String,
        /// Parameters: (name, type).
        params: Vec<(String, String)>,
        /// Return columns: (name, type).
        returns: Vec<(String, String)>,
        /// Raw GQL body.
        body: String,
    },

    /// Drop a stored procedure.
    DropProcedure {
        /// Procedure name.
        name: String,
    },

    // === RDF Records ===
    /// Reserved invalid wire tag 15; retained to keep later current tags stable.
    Reserved15,

    /// Reserved invalid wire tag 16; retained to keep later current tags stable.
    Reserved16,

    /// Reserved invalid wire tag 17; retained to keep later current tags stable.
    Reserved17,

    /// Reserved invalid wire tag 18; retained to keep later current tags stable.
    Reserved18,

    /// Reserved invalid wire tag 19; retained to keep later current tags stable.
    Reserved19,

    // === Transaction Control ===
    /// Transaction commit.
    TransactionCommit {
        /// Transaction ID.
        transaction_id: TransactionId,
    },

    /// Transaction abort.
    TransactionAbort {
        /// Transaction ID.
        transaction_id: TransactionId,
    },

    /// Checkpoint marker.
    Checkpoint {
        /// Transaction ID at checkpoint.
        transaction_id: TransactionId,
    },

    // === Metadata ===
    /// Marks an epoch boundary in the WAL.
    ///
    /// Logged after each `TransactionCommit` to record the new epoch value.
    /// Used by incremental backup to identify WAL segment boundaries and
    /// by point-in-time recovery to stop replay at a target epoch.
    /// Recovery treats this as metadata (no store mutation).
    EpochAdvance {
        /// The epoch after the commit.
        epoch: EpochId,
    },

    /// Force-synced commit that includes the crash-stable epoch (C1).
    ///
    /// Writers emit this instead of a following [`EpochAdvance`](Self::EpochAdvance).
    Committed {
        /// Transaction ID.
        transaction_id: TransactionId,
        /// Epoch assigned at commit, durable with the commit record.
        epoch: EpochId,
    },

    /// Transaction-owned LPG mutation at one exact native graph path.
    LpgMutation {
        /// Transaction that produced this mutation.
        transaction_id: TransactionId,
        /// Exact literal-component graph coordinate; an empty path is root.
        #[serde(with = "grafeo_common::types::graph_path_bytes")]
        graph: GraphPath,
        /// The LPG operation.
        op: LpgMutationOp,
    },

    /// Reserved invalid wire tag 26; retained to keep later current tags stable.
    Reserved26,

    /// Persisted graph model (0 = LPG, 1 = RDF, 2 = Both). Metadata; always recovered.
    GraphModelMeta {
        /// 0 = LPG, 1 = RDF, 2 = Both.
        model: u8,
    },

    /// Reserved invalid wire tag 28; retained to keep later current tags stable.
    Reserved28,

    /// Create an LPG named graph, tagged so abort does not apply it.
    CreateLpgGraph {
        /// Exact non-root native graph coordinate.
        #[serde(with = "grafeo_common::types::graph_path_bytes")]
        graph: GraphPath,
        /// Exact native named-graph lifetime.
        incarnation: GraphIncarnationId,
        /// Transaction that created the graph.
        transaction_id: TransactionId,
    },

    /// Drop an LPG named graph, tagged so abort does not apply it.
    DropLpgGraph {
        /// Exact non-root native graph coordinate.
        #[serde(with = "grafeo_common::types::graph_path_bytes")]
        graph: GraphPath,
        /// Exact native named-graph lifetime.
        incarnation: GraphIncarnationId,
        /// Transaction that dropped the graph.
        transaction_id: TransactionId,
    },

    /// Reserved invalid wire tag 31; retained to keep later current tags stable.
    Reserved31,

    /// Marks a savepoint inside a transaction's WAL stream.
    ///
    /// Recovery consumes this protocol record;
    /// it is never replayed as a store mutation.
    TransactionSavepoint {
        /// Transaction that owns the savepoint.
        transaction_id: TransactionId,
        /// Session-local savepoint name. The most recent matching marker wins.
        name: String,
    },

    /// Rolls a transaction's WAL stream back to a prior savepoint.
    ///
    /// Recovery removes only records belonging to `transaction_id` at or
    /// after the matching [`TransactionSavepoint`](Self::TransactionSavepoint).
    /// Records from concurrently interleaved transactions are preserved.
    TransactionRollbackToSavepoint {
        /// Transaction whose buffered WAL tail is being discarded.
        transaction_id: TransactionId,
        /// Savepoint name, resolved from the most recent matching marker.
        name: String,
    },

    /// Transactional LPG graph-to-graph-type binding replacement.
    ///
    /// `graph_type = None` removes the binding. It shares the owning graph DDL
    /// transaction, so abort and rollback-to-savepoint filtering apply to the
    /// graph and its catalog binding as one unit.
    SetGraphTypeBinding {
        /// Transaction that owns this binding change.
        transaction_id: TransactionId,
        /// Exact native graph coordinate, independent of language schema names.
        #[serde(with = "grafeo_common::types::graph_path_bytes")]
        graph: GraphPath,
        /// Replacement graph type, or `None` to remove the binding.
        graph_type: Option<String>,
    },

    /// Atomically framed standalone catalog statement.
    ///
    /// Older writers emitted each schema side effect as a separate untagged
    /// metadata record, so a multi-effect statement could be torn by a crash or
    /// late validation error. Schema effects can be grouped inside one
    /// checksummed WAL frame. Transaction-owned effects are forbidden. The batch is
    /// metadata (not transaction-owned); recovery either decodes the complete
    /// frame or ignores its truncated tail.
    CatalogBatchV2 {
        /// Batch payload version. Version 1 contains standalone schema
        /// records and rejects nested batches during replay.
        version: u8,
        /// Ordered effects of one catalog statement.
        #[serde(deserialize_with = "deserialize_catalog_batch_records")]
        records: Vec<WalRecord>,
    },

    /// Reserved invalid wire tag 36; retained to keep later current tags stable.
    Reserved36,

    /// Reserved invalid wire tag 37; retained to keep later current tags stable.
    Reserved37,

    /// Epoch-framed, complete post-image of one standalone catalog statement.
    ///
    /// Version 2's
    /// opaque `catalog_state` payload is owned and decoded by the engine; the
    /// storage layer only provides atomic framing, durability classification,
    /// and point-in-time epoch selection.
    CatalogBatchV3 {
        /// Payload format version. Version 2 is a complete engine catalog
        /// post-image, not a sequence of replay deltas.
        version: u8,
        /// Publication epoch reserved for this standalone catalog change.
        epoch: EpochId,
        /// Opaque, engine-owned complete catalog post-image.
        catalog_state: Vec<u8>,
        /// Named LPG graph partitions created by this publication.
        #[serde(with = "graph_paths")]
        created_graphs: Vec<GraphPath>,
        /// Named LPG graph partitions dropped by this publication.
        #[serde(with = "graph_paths")]
        dropped_graphs: Vec<GraphPath>,
        /// Native lifetimes in exactly the same order as created_graphs.
        created_graph_incarnations: Vec<GraphIncarnationId>,
        /// Native lifetimes in exactly the same order as dropped_graphs.
        dropped_graph_incarnations: Vec<GraphIncarnationId>,
    },

    /// Reserved invalid wire tag 39; retained to keep later current tags stable.
    Reserved39,

    /// Reserved invalid wire tag 40; retained to keep later current tags stable.
    Reserved40,

    /// Reserved invalid wire tag 41; retained to keep later current tags stable.
    Reserved41,

    /// Canonical logical-store identity and truthful RDF history boundary.
    ///
    /// A new persistent database writes and force-syncs this record before it
    /// can publish mutations. Recovery treats it as standalone metadata.
    StoreIdentityMeta {
        /// Exact versioned identity metadata shared with snapshots/containers.
        metadata: WorldIdentityMetadataV1,
    },

    /// Transactional creation of one exact RDF named-graph incarnation.
    ///
    /// Unlike the legacy name-only record, replay never allocates a new
    /// process-dependent incarnation.
    CreateNamedRdfGraphV2 {
        /// Named graph IRI.
        name: String,
        /// Exact durable incarnation allocated by the originating store.
        incarnation: GraphIncarnationId,
        /// Transaction that created the graph.
        transaction_id: TransactionId,
    },

    /// Transactional removal of one exact RDF named-graph incarnation.
    DropNamedRdfGraphV2 {
        /// Named graph IRI.
        name: String,
        /// Exact durable incarnation that must currently own `name`.
        incarnation: GraphIncarnationId,
        /// Transaction that dropped the graph.
        transaction_id: TransactionId,
    },

    /// Standalone durable high-water mark for RDF graph incarnations.
    ///
    /// This metadata is force-synced independently of transaction outcome, so
    /// an aborted graph reservation remains a permanent gap after reopen.
    RdfGraphIncarnationHighWaterMeta {
        /// Logical store whose allocator this metadata belongs to.
        store_id: StoreId,
        /// First incarnation that has not yet been reserved.
        next_incarnation: GraphIncarnationId,
    },

    /// Typed RDF quad assertion carrying its exact graph incarnation.
    ///
    /// The paired optional bounds are signed TAI nanoseconds. They are both
    /// absent for an assertion without application valid-time.
    InsertRdfQuadV3 {
        /// Subject term (N-Triples encoding).
        subject: String,
        /// Predicate term (N-Triples encoding).
        predicate: String,
        /// Object term (N-Triples encoding).
        object: String,
        /// Target graph name (`None` = default graph).
        graph: Option<String>,
        /// Exact target graph incarnation (`0` for the default graph).
        graph_incarnation: GraphIncarnationId,
        /// Inclusive valid-time start.
        valid_from_tai_ns: Option<i128>,
        /// Exclusive valid-time end.
        valid_to_tai_ns: Option<i128>,
        /// Transaction that produced this assertion.
        transaction_id: TransactionId,
    },

    /// Typed RDF quad retraction carrying its exact graph incarnation.
    DeleteRdfQuadV3 {
        /// Subject term (N-Triples encoding).
        subject: String,
        /// Predicate term (N-Triples encoding).
        predicate: String,
        /// Object term (N-Triples encoding).
        object: String,
        /// Target graph name (`None` = default graph).
        graph: Option<String>,
        /// Exact target graph incarnation (`0` for the default graph).
        graph_incarnation: GraphIncarnationId,
        /// Transaction that produced this retraction.
        transaction_id: TransactionId,
    },

    /// Complete declaration of one versioned RDF→LPG mapping.
    ///
    /// This remains a standalone, epoch-framed metadata publication. The full
    /// mapping digest is authoritative; `projection_id` is retained only as a
    /// compatibility shorthand and collisions are rejected by the engine.
    RdfLpgProjectionDeclaredV3 {
        /// Compatibility projection id.
        projection_id: u64,
        /// Full digest of the canonical logical mapping.
        mapping_digest: Digest256,
        /// Canonical mapping grammar version.
        mapping_format_version: u16,
        /// Logical source graph (`None` is the default graph).
        source_graph: Option<String>,
        /// RDF class IRI selected by the mapping.
        type_iri: String,
        /// LPG node label produced by the mapping.
        node_label: String,
        /// Shared metadata publication epoch.
        epoch: EpochId,
    },

    /// Transaction-owned RDF→LPG publication receipt.
    ///
    /// `receipt` is the projection subsystem's exact bounded V3 wire payload.
    /// The record does not sync independently and is not metadata: recovery
    /// exposes it only when the matching [`Committed`](Self::Committed) marker
    /// durably commits the LPG row mutations in the same transaction.
    RdfLpgProjectionPublishedV3 {
        /// Transaction that owns both materialized rows and receipt.
        transaction_id: TransactionId,
        /// Exact canonical receipt bytes.
        receipt: Vec<u8>,
    },
    /// Exact engine-owned index owner changes, visible only with this
    /// transaction's durable commit marker. Payload uses the bounded current
    /// engine commit envelope (generation 3): logical owner codec plus exact
    /// physical postimages. It is not an independently committed catalog batch.
    IndexOwnerBatch {
        /// Transaction that owns these logical and physical changes.
        transaction_id: TransactionId,
        /// Owner descriptors, floors, publication epochs and postimages (at most 16 MiB).
        payload: Vec<u8>,
    },
    /// Current engine catalog metadata, owned by one transaction. Only its
    /// durable Committed marker makes this postimage recoverable.
    CatalogPostimage {
        /// Non-SYSTEM transaction that owns the metadata and data changes.
        transaction_id: TransactionId,
        /// Bounded current catalog image retaining the index owner preimage.
        payload: Vec<u8>,
    },
    /// Engine-owned canonical model-local feed, authenticated with native state.
    CdcBatch {
        /// Non-SYSTEM transaction that staged the events.
        transaction_id: TransactionId,
        /// Final commit epoch, resolved before serialization.
        epoch: EpochId,
        /// Native model: 1 = LPG, 2 = RDF.
        model: u8,
        /// Bounded canonical engine payload (at most 16 MiB).
        payload: Vec<u8>,
    },
    /// Commit marker authenticating an exact set of model-local feed batches.
    CommittedWithCdc {
        /// Owning transaction.
        transaction_id: TransactionId,
        /// Final native state and feed epoch.
        epoch: EpochId,
        /// Bitmask of nonempty model batches (LPG=1, RDF=2); zero is empty.
        models: u8,
    },
    /// Durable removal of an epoch-aligned prefix of the retained CDC window.
    /// Does not authorize retirement of graph-recovery WAL segments.
    CdcRetention {
        /// Committed database cut at which retention ran.
        epoch: EpochId,
        /// Feed generation shared by the preimage and postimage.
        generation: u64,
        /// First retained sequence before publication.
        previous_floor: u64,
        /// First retained sequence after publication.
        floor: u64,
        /// Exclusive sequence high-water, unchanged by retention.
        next_sequence: u64,
    },
}

/// LPG data mutation carried by [`WalRecord::LpgMutation`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LpgMutationOp {
    /// Create a node.
    CreateNode {
        /// Node ID.
        id: NodeId,
        /// Labels.
        labels: Vec<String>,
    },
    /// Delete a node.
    DeleteNode {
        /// Node ID.
        id: NodeId,
    },
    /// Create an edge.
    CreateEdge {
        /// Edge ID.
        id: EdgeId,
        /// Source.
        src: NodeId,
        /// Destination.
        dst: NodeId,
        /// Type.
        edge_type: String,
    },
    /// Delete an edge.
    DeleteEdge {
        /// Edge ID.
        id: EdgeId,
    },
    /// Set a node property.
    SetNodeProperty {
        /// Node ID.
        id: NodeId,
        /// Key.
        key: String,
        /// Value.
        value: Value,
    },
    /// Set an edge property.
    SetEdgeProperty {
        /// Edge ID.
        id: EdgeId,
        /// Key.
        key: String,
        /// Value.
        value: Value,
    },
    /// Remove a node property.
    RemoveNodeProperty {
        /// Node ID.
        id: NodeId,
        /// Key.
        key: String,
    },
    /// Remove an edge property.
    RemoveEdgeProperty {
        /// Edge ID.
        id: EdgeId,
        /// Key.
        key: String,
    },
    /// Add a node label.
    AddNodeLabel {
        /// Node ID.
        id: NodeId,
        /// Label.
        label: String,
    },
    /// Remove a node label.
    RemoveNodeLabel {
        /// Node ID.
        id: NodeId,
        /// Label.
        label: String,
    },
    /// Exact ordered label images at the enclosing transaction's commit epoch.
    NodeLabelImages {
        /// Node ID in this record's exact graph incarnation.
        id: NodeId,
        /// The first image replaces this transaction's node creation intent.
        birth: bool,
        /// Complete label sets. Order and repeated images are significant.
        images: Vec<Vec<String>>,
    },
    /// Publishes the enclosing commit epoch for an existing named graph.
    ///
    /// Carries no entity or history mutation. Requires a non-root path
    /// and a non-SYSTEM transaction; the root uses `Committed`.
    PublishGraph,
}

impl WalRecord {
    /// Transaction-owned LPG mutation carrying one exact native graph path.
    #[must_use]
    pub fn lpg(transaction_id: TransactionId, graph: GraphPath, op: LpgMutationOp) -> Self {
        Self::LpgMutation {
            transaction_id,
            graph,
            op,
        }
    }
}

impl WalEntry for WalRecord {
    fn validate_recovery(&self) -> std::result::Result<(), String> {
        fn reject_pending(epoch: EpochId, coordinate: &str) -> std::result::Result<(), String> {
            if epoch == EpochId::PENDING {
                return Err(format!(
                    "{coordinate} uses the reserved PENDING epoch sentinel"
                ));
            }
            Ok(())
        }

        fn require_publication_epoch(
            epoch: EpochId,
            coordinate: &str,
        ) -> std::result::Result<(), String> {
            reject_pending(epoch, coordinate)?;
            if epoch == EpochId::INITIAL {
                return Err(format!("{coordinate} must be greater than epoch zero"));
            }
            Ok(())
        }

        if self.transaction_id().is_some_and(|id| !id.is_valid()) {
            return Err("transaction-bearing WAL record uses TransactionId::INVALID".to_string());
        }

        match self {
            WalRecord::IndexOwnerBatch {
                transaction_id,
                payload,
            }
            | WalRecord::CatalogPostimage {
                transaction_id,
                payload,
            } => {
                if *transaction_id == TransactionId::SYSTEM
                    || payload.is_empty()
                    || payload.len() > 16 * 1024 * 1024
                {
                    return Err("owner batch requires a non-SYSTEM transaction and bounded nonempty payload".into());
                }
                Ok(())
            }
            WalRecord::CreateLpgGraph {
                graph, incarnation, ..
            }
            | WalRecord::DropLpgGraph {
                graph, incarnation, ..
            } if graph.components().is_empty()
                || incarnation.is_default_graph()
                || incarnation.as_u64() == u64::MAX =>
            {
                Err(
                    "LPG graph lifecycle requires a non-root path and valid named incarnation"
                        .into(),
                )
            }
            WalRecord::LpgMutation {
                transaction_id,
                graph,
                op: LpgMutationOp::PublishGraph,
            } => {
                if graph.components().is_empty() || *transaction_id == TransactionId::SYSTEM {
                    return Err(
                        "WAL graph publication requires a named graph and non-SYSTEM transaction"
                            .into(),
                    );
                }
                Ok(())
            }
            WalRecord::LpgMutation {
                transaction_id,
                op: LpgMutationOp::NodeLabelImages { id, images, .. },
                ..
            } => {
                if *transaction_id == TransactionId::SYSTEM {
                    return Err("WAL label images require a non-SYSTEM transaction".into());
                }
                if !id.is_valid() || images.is_empty() {
                    return Err("invalid node identity or empty WAL label-image sequence".into());
                }
                for labels in images {
                    if labels.len() > usize::from(u16::MAX) {
                        return Err("WAL label image exceeds label capacity".into());
                    }
                    let mut unique = std::collections::HashSet::with_capacity(labels.len());
                    if labels.iter().any(|label| !unique.insert(label.as_str())) {
                        return Err("WAL label image contains duplicate labels".into());
                    }
                }
                Ok(())
            }
            // Exact-history WAL-directory saves emit a synthetic committed
            // group at the original epoch zero.
            WalRecord::EpochAdvance { epoch } => reject_pending(*epoch, "EpochAdvance.epoch"),
            WalRecord::Committed { epoch, .. } => reject_pending(*epoch, "Committed.epoch"),
            WalRecord::CdcBatch {
                transaction_id,
                epoch,
                model,
                payload,
            } => {
                require_publication_epoch(*epoch, "CdcBatch.epoch")?;
                if *transaction_id == TransactionId::SYSTEM
                    || !matches!(model, 1 | 2)
                    || payload.is_empty()
                    || payload.len() > 16 * 1024 * 1024
                {
                    return Err("invalid CDC transaction, model or payload size".into());
                }
                Ok(())
            }
            WalRecord::CommittedWithCdc {
                transaction_id,
                epoch,
                models,
            } => {
                require_publication_epoch(*epoch, "CommittedWithCdc.epoch")?;
                if *transaction_id == TransactionId::SYSTEM || *models & !3 != 0 {
                    return Err("invalid CDC commit transaction or model manifest".into());
                }
                Ok(())
            }
            WalRecord::CdcRetention {
                epoch,
                generation,
                previous_floor,
                floor,
                next_sequence,
            } => {
                reject_pending(*epoch, "CdcRetention.epoch")?;
                if *generation == 0
                    || *generation == u64::MAX
                    || *previous_floor == 0
                    || previous_floor >= floor
                    || floor > next_sequence
                    || *next_sequence == u64::MAX
                {
                    return Err("invalid CDC retention coordinates".into());
                }
                Ok(())
            }
            WalRecord::Reserved15
            | WalRecord::Reserved16
            | WalRecord::Reserved17
            | WalRecord::Reserved18
            | WalRecord::Reserved19
            | WalRecord::Reserved26
            | WalRecord::Reserved28
            | WalRecord::Reserved31
            | WalRecord::Reserved41
            | WalRecord::Reserved36
            | WalRecord::Reserved37
            | WalRecord::Reserved39
            | WalRecord::Reserved40 => Err("reserved WAL record tag".into()),
            WalRecord::CatalogBatchV3 {
                version,
                epoch,
                created_graphs,
                dropped_graphs,
                created_graph_incarnations,
                dropped_graph_incarnations,
                ..
            } => {
                require_publication_epoch(*epoch, "CatalogBatchV3.epoch")?;
                if *version != 2
                    || created_graphs.len() != created_graph_incarnations.len()
                    || dropped_graphs.len() != dropped_graph_incarnations.len()
                    || created_graph_incarnations
                        .iter()
                        .chain(dropped_graph_incarnations)
                        .any(|id| id.is_default_graph() || id.as_u64() == u64::MAX)
                {
                    return Err(
                        "catalog lifecycle requires matching native incarnation coordinates".into(),
                    );
                }
                if created_graphs
                    .iter()
                    .chain(dropped_graphs)
                    .any(|graph| graph.components().is_empty())
                {
                    return Err("catalog lifecycle cannot create or drop the root".into());
                }
                Ok(())
            }
            WalRecord::RdfLpgProjectionDeclaredV3 { epoch, .. } => {
                require_publication_epoch(*epoch, "RdfLpgProjectionDeclaredV3.epoch")
            }
            WalRecord::CatalogBatchV2 { records, .. } => {
                for record in records {
                    if matches!(record, WalRecord::CatalogBatchV2 { .. }) {
                        return Err(NESTED_CATALOG_BATCH_DECODE_ERROR.to_string());
                    }
                    if !matches!(
                        record,
                        WalRecord::CreateNodeType { .. }
                            | WalRecord::DropNodeType { .. }
                            | WalRecord::CreateEdgeType { .. }
                            | WalRecord::DropEdgeType { .. }
                            | WalRecord::CreateConstraint { .. }
                            | WalRecord::DropConstraint { .. }
                            | WalRecord::CreateGraphType { .. }
                            | WalRecord::DropGraphType { .. }
                            | WalRecord::CreateSchema { .. }
                            | WalRecord::DropSchema { .. }
                            | WalRecord::AlterNodeType { .. }
                            | WalRecord::AlterEdgeType { .. }
                            | WalRecord::AlterGraphType { .. }
                            | WalRecord::CreateProcedure { .. }
                            | WalRecord::DropProcedure { .. }
                    ) {
                        return Err("CatalogBatchV2 permits only standalone schema effects".into());
                    }
                    record.validate_recovery()?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn validate_commit_group<'a>(
        &self,
        members: impl Iterator<Item = &'a Self>,
    ) -> std::result::Result<(), String> {
        let mut found = 0;
        for member in members {
            if let Self::CdcBatch {
                transaction_id,
                epoch,
                model,
                ..
            } = member
            {
                member.validate_recovery()?;
                let Self::CommittedWithCdc {
                    transaction_id: owner,
                    epoch: committed,
                    ..
                } = self
                else {
                    return Err("CDC batch has no CDC commit manifest".into());
                };
                if transaction_id != owner || epoch != committed || found >= *model {
                    return Err("duplicate or foreign CDC batch coordinate".into());
                }
                found |= model;
            }
        }
        if let Self::CommittedWithCdc { models, .. } = self
            && found != *models
        {
            return Err("CDC commit manifest does not match its batches".into());
        }
        Ok(())
    }

    fn commit_epoch(&self) -> Option<EpochId> {
        match self {
            Self::Committed { epoch, .. } | Self::CommittedWithCdc { epoch, .. } => Some(*epoch),
            _ => None,
        }
    }

    fn feed_epoch(&self) -> Option<EpochId> {
        match self {
            Self::CdcBatch { epoch, .. } | Self::CommittedWithCdc { epoch, .. } => Some(*epoch),
            _ => None,
        }
    }

    fn requires_sync(&self) -> bool {
        matches!(
            self,
            WalRecord::TransactionCommit { .. }
                | WalRecord::Committed { .. }
                | WalRecord::CommittedWithCdc { .. }
                | WalRecord::TransactionAbort { .. }
                | WalRecord::CdcRetention { .. }
                | WalRecord::CatalogBatchV2 { .. }
                | WalRecord::CatalogBatchV3 { .. }
                | WalRecord::RdfLpgProjectionDeclaredV3 { .. }
                | WalRecord::StoreIdentityMeta { .. }
                | WalRecord::RdfGraphIncarnationHighWaterMeta { .. }
        )
    }

    fn is_commit(&self) -> bool {
        matches!(
            self,
            WalRecord::TransactionCommit { .. }
                | WalRecord::Committed { .. }
                | WalRecord::CommittedWithCdc { .. }
        )
    }

    fn transaction_id(&self) -> Option<TransactionId> {
        match self {
            WalRecord::InsertRdfQuadV3 { transaction_id, .. }
            | WalRecord::DeleteRdfQuadV3 { transaction_id, .. }
            | WalRecord::TransactionCommit { transaction_id }
            | WalRecord::TransactionAbort { transaction_id }
            | WalRecord::Checkpoint { transaction_id }
            | WalRecord::Committed { transaction_id, .. }
            | WalRecord::CommittedWithCdc { transaction_id, .. }
            | WalRecord::CdcBatch { transaction_id, .. }
            | WalRecord::LpgMutation { transaction_id, .. }
            | WalRecord::DropNamedRdfGraphV2 { transaction_id, .. }
            | WalRecord::CreateNamedRdfGraphV2 { transaction_id, .. }
            | WalRecord::CreateLpgGraph { transaction_id, .. }
            | WalRecord::DropLpgGraph { transaction_id, .. }
            | WalRecord::TransactionSavepoint { transaction_id, .. }
            | WalRecord::TransactionRollbackToSavepoint { transaction_id, .. }
            | WalRecord::IndexOwnerBatch { transaction_id, .. }
            | WalRecord::CatalogPostimage { transaction_id, .. }
            | WalRecord::SetGraphTypeBinding { transaction_id, .. } => Some(*transaction_id),
            WalRecord::RdfLpgProjectionPublishedV3 { transaction_id, .. } => Some(*transaction_id),
            _ => None,
        }
    }

    fn savepoint_name(&self) -> Option<&str> {
        match self {
            WalRecord::TransactionSavepoint { name, .. } => Some(name),
            _ => None,
        }
    }

    fn rollback_to_savepoint_name(&self) -> Option<&str> {
        match self {
            WalRecord::TransactionRollbackToSavepoint { name, .. } => Some(name),
            _ => None,
        }
    }

    fn is_abort(&self) -> bool {
        matches!(self, WalRecord::TransactionAbort { .. })
    }

    fn is_checkpoint(&self) -> bool {
        matches!(self, WalRecord::Checkpoint { .. })
    }

    fn is_catalog_batch(&self) -> bool {
        matches!(self, WalRecord::CatalogBatchV3 { .. })
    }

    fn is_lpg_mutation(&self) -> bool {
        matches!(self, WalRecord::LpgMutation { .. })
    }

    fn is_data_mutation(&self) -> bool {
        matches!(
            self,
            WalRecord::LpgMutation { .. }
                | WalRecord::InsertRdfQuadV3 { .. }
                | WalRecord::DeleteRdfQuadV3 { .. }
                | WalRecord::CreateNamedRdfGraphV2 { .. }
                | WalRecord::DropNamedRdfGraphV2 { .. }
                | WalRecord::RdfLpgProjectionPublishedV3 { .. }
        )
    }

    fn is_metadata(&self) -> bool {
        matches!(
            self,
            WalRecord::EpochAdvance { .. }
                | WalRecord::CdcRetention { .. }
                | WalRecord::GraphModelMeta { .. }
                // Standalone catalog DDL is auto-commit structural state.
                // Treating it as metadata keeps a concurrent abort from stealing
                // the record. In-transaction graph create/drop uses the tagged
                // `CreateLpgGraph` / `DropLpgGraph` variants instead.
                | WalRecord::CreateNodeType { .. }
                | WalRecord::DropNodeType { .. }
                | WalRecord::CreateEdgeType { .. }
                | WalRecord::DropEdgeType { .. }
                | WalRecord::CreateConstraint { .. }
                | WalRecord::DropConstraint { .. }
                | WalRecord::CreateGraphType { .. }
                | WalRecord::DropGraphType { .. }
                | WalRecord::CreateSchema { .. }
                | WalRecord::DropSchema { .. }
                | WalRecord::AlterNodeType { .. }
                | WalRecord::AlterEdgeType { .. }
                | WalRecord::AlterGraphType { .. }
                | WalRecord::CreateProcedure { .. }
                | WalRecord::DropProcedure { .. }
                | WalRecord::CatalogBatchV2 { .. }
                | WalRecord::CatalogBatchV3 { .. }
                | WalRecord::RdfLpgProjectionDeclaredV3 { .. }
                | WalRecord::StoreIdentityMeta { .. }
                | WalRecord::RdfGraphIncarnationHighWaterMeta { .. }
        )
    }

    fn make_checkpoint(transaction_id: TransactionId) -> Self {
        WalRecord::Checkpoint { transaction_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdc_retention_is_bounded_standalone_durable_metadata() {
        let valid = WalRecord::CdcRetention {
            epoch: EpochId::new(2),
            generation: 1,
            previous_floor: 1,
            floor: 3,
            next_sequence: 3,
        };
        assert!(valid.validate_recovery().is_ok());
        assert!(valid.is_metadata());
        assert!(valid.requires_sync());
        assert!(!valid.is_checkpoint());
        assert!(!valid.is_commit());
        assert_eq!(valid.transaction_id(), None);
        for (epoch, generation, previous_floor, floor, next_sequence) in [
            (EpochId::PENDING, 1, 1, 3, 3),
            (EpochId::new(2), 0, 1, 3, 3),
            (EpochId::new(2), u64::MAX, 1, 3, 3),
            (EpochId::new(2), 1, 0, 3, 3),
            (EpochId::new(2), 1, 3, 3, 3),
            (EpochId::new(2), 1, 1, 4, 3),
            (EpochId::new(2), 1, 1, 3, u64::MAX),
        ] {
            assert!(
                WalRecord::CdcRetention {
                    epoch,
                    generation,
                    previous_floor,
                    floor,
                    next_sequence
                }
                .validate_recovery()
                .is_err()
            );
        }
    }

    #[test]
    fn owner_batches_are_transaction_owned_and_bounded() {
        transaction_envelope_contract(|transaction_id, payload| WalRecord::IndexOwnerBatch {
            transaction_id,
            payload,
        });
    }

    #[test]
    fn catalog_postimages_are_transaction_owned_and_bounded() {
        transaction_envelope_contract(|transaction_id, payload| WalRecord::CatalogPostimage {
            transaction_id,
            payload,
        });
    }

    fn transaction_envelope_contract(make: impl Fn(TransactionId, Vec<u8>) -> WalRecord) {
        use crate::wal::{WalManager, WalRecovery};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = WalManager::with_config(&path, Default::default()).unwrap();
        let tx = TransactionId::new(9);
        let record = make(tx, vec![1]);
        assert_eq!(record.transaction_id(), Some(tx));
        assert!(!record.is_metadata());
        assert!(!record.is_catalog_batch());
        wal.log(&record).unwrap();
        wal.log(&WalRecord::TransactionAbort { transaction_id: tx })
            .unwrap();
        let tx = TransactionId::new(10);
        wal.log(&WalRecord::TransactionSavepoint {
            transaction_id: tx,
            name: "s".into(),
        })
        .unwrap();
        wal.log(&make(tx, vec![2])).unwrap();
        wal.log(&WalRecord::TransactionRollbackToSavepoint {
            transaction_id: tx,
            name: "s".into(),
        })
        .unwrap();
        wal.log(&make(tx, vec![3])).unwrap();
        wal.log(&WalRecord::Committed {
            transaction_id: tx,
            epoch: EpochId::new(1),
        })
        .unwrap();
        wal.sync().unwrap();
        drop(wal);
        let records = WalRecovery::new(&path).unwrap().recover().unwrap();
        let payloads: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                WalRecord::IndexOwnerBatch { payload, .. }
                | WalRecord::CatalogPostimage { payload, .. } => Some(payload.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(payloads, vec![vec![3]]);
        for (transaction_id, payload) in [
            (TransactionId::SYSTEM, vec![1]),
            (tx, vec![]),
            (tx, vec![0; 16 * 1024 * 1024 + 1]),
        ] {
            assert!(make(transaction_id, payload).validate_recovery().is_err());
        }
    }

    fn roundtrip(record: &WalRecord) -> WalRecord {
        let json = serde_json::to_string(record).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn current_rdf_records_retain_exact_parent_bytes() {
        use grafeo_common::types::HistoryCompleteness;

        macro_rules! captured {
            ($name:literal) => {
                (
                    $name,
                    include_bytes!(concat!(
                        "../../tests/fixtures/rdf-wal-predecessors/",
                        $name,
                        ".body.bin"
                    ))
                    .as_slice(),
                    include_bytes!(concat!(
                        "../../tests/fixtures/rdf-wal-predecessors/",
                        $name,
                        ".payload.bin"
                    ))
                    .as_slice(),
                )
            };
        }
        let check = |record: WalRecord, tag: u8, (name, body, payload): (&str, &[u8], &[u8])| {
            assert_eq!(body[0], tag, "{name}");
            record.validate_recovery().unwrap();
            assert_eq!(
                bincode::serde::encode_to_vec(&record, bincode::config::standard()).unwrap(),
                body,
                "{name}"
            );
            assert_eq!(
                crate::wal::encode_record(&record).unwrap(),
                payload,
                "{name}"
            );
            let (decoded, consumed): (WalRecord, usize) =
                bincode::serde::decode_from_slice(body, bincode::config::standard()).unwrap();
            assert_eq!(consumed, body.len(), "{name}");
            assert_eq!(decoded.transaction_id(), record.transaction_id(), "{name}");
            assert_eq!(
                crate::wal::encode_record(&decoded).unwrap(),
                payload,
                "{name}"
            );
        };
        let transaction_id = TransactionId::new(9);
        for (graph, graph_incarnation, insert, delete) in [
            (
                None,
                GraphIncarnationId::DEFAULT_GRAPH,
                captured!("current-46-insert-default"),
                captured!("current-47-delete-default"),
            ),
            (
                Some("urn:g".to_owned()),
                GraphIncarnationId::new(7),
                captured!("current-46-insert-named"),
                captured!("current-47-delete-named"),
            ),
        ] {
            check(
                WalRecord::InsertRdfQuadV3 {
                    subject: "<urn:s>".into(),
                    predicate: "<urn:p>".into(),
                    object: "\"value\"".into(),
                    graph: graph.clone(),
                    graph_incarnation,
                    valid_from_tai_ns: Some(-9_223_372_036_854_775_809_i128),
                    valid_to_tai_ns: Some(9_223_372_036_854_775_808_i128),
                    transaction_id,
                },
                46,
                insert,
            );
            check(
                WalRecord::DeleteRdfQuadV3 {
                    subject: "<urn:s>".into(),
                    predicate: "<urn:p>".into(),
                    object: "\"value\"".into(),
                    graph,
                    graph_incarnation,
                    transaction_id,
                },
                47,
                delete,
            );
        }
        let store_id = StoreId::from_bytes([0x5a; StoreId::LEN]).unwrap();
        check(
            WalRecord::StoreIdentityMeta {
                metadata: WorldIdentityMetadataV1::new(store_id, HistoryCompleteness::Complete)
                    .unwrap(),
            },
            42,
            captured!("current-42-identity"),
        );
        check(
            WalRecord::CreateNamedRdfGraphV2 {
                name: "urn:g".into(),
                incarnation: GraphIncarnationId::new(7),
                transaction_id,
            },
            43,
            captured!("current-43-create-named"),
        );
        check(
            WalRecord::DropNamedRdfGraphV2 {
                name: "urn:g".into(),
                incarnation: GraphIncarnationId::new(7),
                transaction_id,
            },
            44,
            captured!("current-44-drop-named"),
        );
        check(
            WalRecord::RdfGraphIncarnationHighWaterMeta {
                store_id,
                next_incarnation: GraphIncarnationId::new(8),
            },
            45,
            captured!("current-45-high-water"),
        );
        check(
            WalRecord::RdfLpgProjectionDeclaredV3 {
                projection_id: 17,
                mapping_digest: Digest256::from_bytes([0x44; 32]),
                mapping_format_version: 2,
                source_graph: Some("urn:g".into()),
                type_iri: "urn:Person".into(),
                node_label: "Person".into(),
                epoch: EpochId::new(23),
            },
            48,
            captured!("current-48-projection-declared"),
        );
        check(
            WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id,
                receipt: vec![0x47, 0x52, 0x50, 0x52, 3, 0],
            },
            49,
            captured!("current-49-projection-published"),
        );
    }

    #[test]
    fn standalone_catalog_batches_refuse_transaction_owned_effects() {
        let transaction_id = TransactionId::new(3);
        let graph = GraphPath::from_components(&["a", "", "b/c"]).unwrap();
        for record in [
            WalRecord::lpg(
                transaction_id,
                graph.clone(),
                LpgMutationOp::DeleteNode { id: NodeId::new(1) },
            ),
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: graph.clone(),
                transaction_id,
            },
            WalRecord::DropLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: graph.clone(),
                transaction_id,
            },
            WalRecord::SetGraphTypeBinding {
                transaction_id,
                graph,
                graph_type: Some("T".into()),
            },
            WalRecord::IndexOwnerBatch {
                transaction_id,
                payload: vec![1],
            },
            WalRecord::CatalogPostimage {
                transaction_id,
                payload: vec![1],
            },
            WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![],
                dropped_graph_incarnations: vec![],
                version: 2,
                epoch: EpochId::new(1),
                catalog_state: vec![],
                created_graphs: vec![],
                dropped_graphs: vec![],
            },
        ] {
            let batch = WalRecord::CatalogBatchV2 {
                version: 1,
                records: vec![record],
            };
            assert!(batch.validate_recovery().is_err());
            assert!(crate::wal::encode_record(&batch).is_err());
        }
    }

    #[test]
    fn json_graph_paths_validate_unknown_length_sequences() {
        let record = WalRecord::lpg(
            TransactionId::new(1),
            GraphPath::root(),
            LpgMutationOp::DeleteNode { id: NodeId::new(1) },
        );
        let mut json = serde_json::to_value(record).unwrap();
        for bytes in [vec![0; 3], vec![0, 0, 0, 0, 1], vec![1, 1, 0, 0]] {
            json["LpgMutation"]["graph"] = serde_json::json!(bytes);
            assert!(serde_json::from_value::<WalRecord>(json.clone()).is_err());
            assert!(
                serde_json::from_str::<WalRecord>(&serde_json::to_string(&json).unwrap()).is_err()
            );
        }
    }

    #[test]
    fn test_create_node_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string(), "Employee".to_string()],
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::CreateNode { id, labels },
                ..
            } => {
                assert_eq!(id, NodeId::new(1));
                assert_eq!(labels, vec!["Person", "Employee"]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delete_node_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::DeleteNode {
                id: NodeId::new(42),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::DeleteNode { id },
                ..
            } => assert_eq!(id, NodeId::new(42)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_edge_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateEdge {
                id: EdgeId::new(10),
                src: NodeId::new(1),
                dst: NodeId::new(2),
                edge_type: "KNOWS".to_string(),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op:
                    crate::wal::LpgMutationOp::CreateEdge {
                        id,
                        src,
                        dst,
                        edge_type,
                    },
                ..
            } => {
                assert_eq!(id, EdgeId::new(10));
                assert_eq!(src, NodeId::new(1));
                assert_eq!(dst, NodeId::new(2));
                assert_eq!(edge_type, "KNOWS");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delete_edge_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::DeleteEdge {
                id: EdgeId::new(99),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::DeleteEdge { id },
                ..
            } => assert_eq!(id, EdgeId::new(99)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_node_property_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(5),
                key: "name".to_string(),
                value: Value::String("Alix".into()),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { id, key, value },
                ..
            } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "name");
                assert_eq!(value, Value::String("Alix".into()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_edge_property_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetEdgeProperty {
                id: EdgeId::new(7),
                key: "weight".to_string(),
                value: Value::Float64(std::f64::consts::PI),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetEdgeProperty { id, key, value },
                ..
            } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
                assert_eq!(value, Value::Float64(std::f64::consts::PI));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_node_property_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::RemoveNodeProperty {
                id: NodeId::new(5),
                key: "age".to_string(),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::RemoveNodeProperty { id, key },
                ..
            } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "age");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_edge_property_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::RemoveEdgeProperty {
                id: EdgeId::new(7),
                key: "weight".to_string(),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::RemoveEdgeProperty { id, key },
                ..
            } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_add_node_label_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::AddNodeLabel {
                id: NodeId::new(3),
                label: "Admin".to_string(),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::AddNodeLabel { id, label },
                ..
            } => {
                assert_eq!(id, NodeId::new(3));
                assert_eq!(label, "Admin");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_node_label_roundtrip() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::RemoveNodeLabel {
                id: NodeId::new(3),
                label: "Temp".to_string(),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::RemoveNodeLabel { id, label },
                ..
            } => {
                assert_eq!(id, NodeId::new(3));
                assert_eq!(label, "Temp");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_commit_roundtrip() {
        let record = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(100),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TransactionCommit { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(100));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_abort_roundtrip() {
        let record = WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(200),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TransactionAbort { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(200));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_savepoint_protocol_roundtrip() {
        let savepoint = roundtrip(&WalRecord::TransactionSavepoint {
            transaction_id: TransactionId::new(201),
            name: "stable".to_string(),
        });
        assert!(matches!(
            savepoint,
            WalRecord::TransactionSavepoint { transaction_id, ref name }
                if transaction_id == TransactionId::new(201) && name == "stable"
        ));

        let rollback = roundtrip(&WalRecord::TransactionRollbackToSavepoint {
            transaction_id: TransactionId::new(201),
            name: "stable".to_string(),
        });
        assert!(matches!(
            rollback,
            WalRecord::TransactionRollbackToSavepoint { transaction_id, ref name }
                if transaction_id == TransactionId::new(201) && name == "stable"
        ));
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let record = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(50),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::Checkpoint { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(50));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_node_empty_labels() {
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(0),
                labels: Vec::new(),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::CreateNode { labels, .. },
                ..
            } => assert!(labels.is_empty()),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_lpg_graph_roundtrip() {
        let graph = GraphPath::from_components(&["analytics", "", "events/2026"]).unwrap();
        let record = WalRecord::CreateLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(1),
            graph: graph.clone(),
            transaction_id: TransactionId::new(7),
        };
        assert!(matches!(roundtrip(&record),
            WalRecord::CreateLpgGraph { graph: actual, transaction_id, incarnation }
                if actual == graph && transaction_id == TransactionId::new(7) && incarnation.as_u64() == 1));
    }

    #[test]
    fn test_drop_lpg_graph_roundtrip() {
        let graph = GraphPath::from_components(&["temp"]).unwrap();
        let record = WalRecord::DropLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(1),
            graph: graph.clone(),
            transaction_id: TransactionId::new(8),
        };
        assert!(matches!(roundtrip(&record),
            WalRecord::DropLpgGraph { graph: actual, transaction_id, incarnation }
                if actual == graph && transaction_id == TransactionId::new(8) && incarnation.as_u64() == 1));
    }

    #[test]
    fn lpg_graph_lifecycle_refuses_root() {
        for record in [
            WalRecord::CreateLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: GraphPath::root(),
                transaction_id: TransactionId::new(1),
            },
            WalRecord::DropLpgGraph {
                incarnation: grafeo_common::types::GraphIncarnationId::new(1),
                graph: GraphPath::root(),
                transaction_id: TransactionId::new(1),
            },
        ] {
            assert!(record.validate_recovery().is_err());
        }
    }

    #[test]
    fn native_graph_lifecycle_refuses_reserved_and_exhausted_incarnations() {
        for incarnation in [
            GraphIncarnationId::DEFAULT_GRAPH,
            GraphIncarnationId::new(u64::MAX),
        ] {
            for record in [
                WalRecord::CreateLpgGraph {
                    graph: GraphPath::from_components(&["named"]).unwrap(),
                    incarnation,
                    transaction_id: TransactionId::new(1),
                },
                WalRecord::DropLpgGraph {
                    graph: GraphPath::from_components(&["named"]).unwrap(),
                    incarnation,
                    transaction_id: TransactionId::new(1),
                },
            ] {
                assert!(record.validate_recovery().is_err());
                assert!(crate::wal::encode_record(&record).is_err());
            }
        }
        for (created_graph_incarnations, dropped_graph_incarnations) in [
            (vec![], vec![]),
            (vec![GraphIncarnationId::DEFAULT_GRAPH], vec![]),
            (vec![GraphIncarnationId::new(u64::MAX)], vec![]),
            (
                vec![GraphIncarnationId::new(1)],
                vec![GraphIncarnationId::new(2)],
            ),
        ] {
            let record = WalRecord::CatalogBatchV3 {
                version: 2,
                epoch: EpochId::new(1),
                catalog_state: vec![1],
                created_graphs: vec![GraphPath::from_components(&["named"]).unwrap()],
                dropped_graphs: vec![],
                created_graph_incarnations,
                dropped_graph_incarnations,
            };
            assert!(record.validate_recovery().is_err());
            assert!(crate::wal::encode_record(&record).is_err());
        }
    }

    #[test]
    fn test_wal_entry_requires_sync() {
        use super::WalEntry;

        // Commit and abort markers force fsync in Sync durability mode.
        assert!(
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::lpg(
                TransactionId::new(1),
                grafeo_common::types::GraphPath::root(),
                crate::wal::LpgMutationOp::CreateNode {
                    id: NodeId::new(1),
                    labels: vec![]
                }
            )
            .requires_sync()
        );

        assert!(
            WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::Checkpoint {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );
    }

    #[test]
    fn test_wal_entry_transaction_markers() {
        use super::WalEntry;

        let commit = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        };
        assert!(commit.is_commit());
        assert!(!commit.is_abort());
        assert!(!commit.is_checkpoint());

        let abort = WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(2),
        };
        assert!(!abort.is_commit());
        assert!(abort.is_abort());
        assert!(!abort.is_checkpoint());

        let checkpoint = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(3),
        };
        assert!(!checkpoint.is_commit());
        assert!(!checkpoint.is_abort());
        assert!(checkpoint.is_checkpoint());

        // Data records are none of the above
        let data = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec![],
            },
        );
        assert!(!data.is_commit());
        assert!(!data.is_abort());
        assert!(!data.is_checkpoint());
    }

    #[test]
    fn test_wal_entry_make_checkpoint() {
        use super::WalEntry;

        let record = WalRecord::make_checkpoint(TransactionId::new(42));
        match record {
            WalRecord::Checkpoint { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(42));
            }
            _ => panic!("make_checkpoint should produce Checkpoint variant"),
        }
    }

    // =========================================================================
    // T1-05: Serialization round-trip for untested Value types
    // =========================================================================

    #[test]
    fn test_value_map_roundtrip() {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("name"), Value::String("Alix".into()));
        map.insert(PropertyKey::from("age"), Value::Int64(30));

        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(1),
                key: "metadata".to_string(),
                value: Value::Map(Arc::new(map.clone())),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Map(m) => {
                    assert_eq!(m.len(), 2);
                    assert_eq!(m[&PropertyKey::from("name")], Value::String("Alix".into()));
                    assert_eq!(m[&PropertyKey::from("age")], Value::Int64(30));
                }
                other => panic!("Expected Map, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_vector_roundtrip() {
        use std::sync::Arc;

        let embedding: Arc<[f32]> = Arc::from(vec![0.1_f32, 0.2, 0.3, 0.4]);
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(2),
                key: "embedding".to_string(),
                value: Value::Vector(embedding.clone()),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Vector(v) => {
                    assert_eq!(v.len(), 4);
                    assert!((v[0] - 0.1).abs() < f32::EPSILON);
                    assert!((v[3] - 0.4).abs() < f32::EPSILON);
                }
                other => panic!("Expected Vector, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_timestamp_roundtrip() {
        use grafeo_common::types::Timestamp;

        let ts = Timestamp::from_secs(1_700_000_000); // 2023-11-14
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(3),
                key: "created_at".to_string(),
                value: Value::Timestamp(ts),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Timestamp(t) => {
                    assert_eq!(t.as_secs(), 1_700_000_000);
                }
                other => panic!("Expected Timestamp, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_zoned_datetime_roundtrip() {
        use grafeo_common::types::{Timestamp, ZonedDatetime};

        let ts = Timestamp::from_secs(1_700_000_000);
        let zdt = ZonedDatetime::from_timestamp_offset(ts, 3600); // +01:00
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(4),
                key: "event_time".to_string(),
                value: Value::ZonedDatetime(zdt),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::ZonedDatetime(z) => {
                    assert_eq!(z.as_timestamp().as_secs(), 1_700_000_000);
                    assert_eq!(z.offset_seconds(), 3600);
                }
                other => panic!("Expected ZonedDatetime, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_path_roundtrip() {
        use std::sync::Arc;

        let nodes: Arc<[Value]> = Arc::from(vec![
            Value::String("node_A".into()),
            Value::String("node_B".into()),
            Value::String("node_C".into()),
        ]);
        let edges: Arc<[Value]> = Arc::from(vec![
            Value::String("edge_AB".into()),
            Value::String("edge_BC".into()),
        ]);
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(5),
                key: "route".to_string(),
                value: Value::Path {
                    nodes: nodes.clone(),
                    edges: edges.clone(),
                },
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Path { nodes: n, edges: e } => {
                    assert_eq!(n.len(), 3);
                    assert_eq!(e.len(), 2);
                    assert_eq!(n[0], Value::String("node_A".into()));
                    assert_eq!(e[1], Value::String("edge_BC".into()));
                }
                other => panic!("Expected Path, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_date_roundtrip() {
        use grafeo_common::types::Date;

        let date = Date::from_ymd(2024, 6, 15).unwrap();
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(6),
                key: "birthday".to_string(),
                value: Value::Date(date),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Date(d) => {
                    assert_eq!(d.year(), 2024);
                    assert_eq!(d.month(), 6);
                    assert_eq!(d.day(), 15);
                }
                other => panic!("Expected Date, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_time_roundtrip() {
        use grafeo_common::types::Time;

        let time = Time::from_hms(14, 30, 45).unwrap().with_offset(3600);
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(7),
                key: "alarm".to_string(),
                value: Value::Time(time),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Time(t) => {
                    assert_eq!(t.hour(), 14);
                    assert_eq!(t.minute(), 30);
                    assert_eq!(t.second(), 45);
                    assert_eq!(t.offset_seconds(), Some(3600));
                }
                other => panic!("Expected Time, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_duration_roundtrip() {
        use grafeo_common::types::Duration;

        let dur = Duration::new(14, 3, 4 * 3_600_000_000_000 + 5 * 60_000_000_000); // P1Y2M3DT4H5M
        let record = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::SetNodeProperty {
                id: NodeId::new(8),
                key: "interval".to_string(),
                value: Value::Duration(dur),
            },
        );
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::LpgMutation {
                op: crate::wal::LpgMutationOp::SetNodeProperty { value, .. },
                ..
            } => match value {
                Value::Duration(d) => {
                    assert_eq!(d.months(), 14);
                    assert_eq!(d.days(), 3);
                }
                other => panic!("Expected Duration, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_epoch_advance_roundtrip() {
        let record = WalRecord::EpochAdvance {
            epoch: EpochId::new(42),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::EpochAdvance { epoch } => assert_eq!(epoch, EpochId::new(42)),
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_epoch_advance_is_metadata() {
        let record = WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        };
        assert!(!record.requires_sync());
        assert!(!record.is_commit());
        assert!(!record.is_abort());
        assert!(!record.is_checkpoint());
        assert!(record.is_metadata());
    }

    #[test]
    fn store_identity_metadata_round_trips_and_forces_durability() {
        use grafeo_common::types::{HistoryCompleteness, StoreId, WorldIdentityMetadataV1};

        let store_id = StoreId::from_bytes([0x5a; StoreId::LEN]).unwrap();
        let metadata =
            WorldIdentityMetadataV1::new(store_id, HistoryCompleteness::Complete).unwrap();
        let record = WalRecord::StoreIdentityMeta {
            metadata: metadata.clone(),
        };

        assert!(record.requires_sync());
        assert!(record.is_metadata());
        assert!(record.transaction_id().is_none());
        match roundtrip(&record) {
            WalRecord::StoreIdentityMeta { metadata: actual } => {
                assert_eq!(actual, metadata);
            }
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_non_metadata_records() {
        let commit = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        };
        assert!(!commit.is_metadata());

        let create = WalRecord::lpg(
            grafeo_common::types::TransactionId::new(1),
            grafeo_common::types::GraphPath::root(),
            crate::wal::LpgMutationOp::CreateNode {
                id: NodeId::new(1),
                labels: vec![],
            },
        );
        assert!(!create.is_metadata());
    }

    #[test]
    fn test_untagged_schema_ddl_is_metadata() {
        let rec = WalRecord::CreateSchema { name: "g".into() };
        assert!(rec.is_metadata());
        assert!(rec.transaction_id().is_none());
    }

    #[test]
    fn test_tagged_lpg_graph_ddl_carries_tid() {
        let rec = WalRecord::CreateLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(1),
            graph: GraphPath::from_components(&["g"]).unwrap(),
            transaction_id: TransactionId::new(7),
        };
        assert_eq!(rec.transaction_id(), Some(TransactionId::new(7)));
        assert!(!rec.is_metadata());
        let drop = WalRecord::DropLpgGraph {
            incarnation: grafeo_common::types::GraphIncarnationId::new(1),
            graph: GraphPath::from_components(&["g"]).unwrap(),
            transaction_id: TransactionId::new(7),
        };
        assert_eq!(drop.transaction_id(), Some(TransactionId::new(7)));
    }

    #[test]
    fn exact_lpg_graph_coordinate_round_trips_without_aliases() {
        let transaction_id = TransactionId::new(71);
        let paths = [
            GraphPath::root(),
            GraphPath::from_components(&[""]).unwrap(),
            GraphPath::from_components(&["a", "b"]).unwrap(),
            GraphPath::from_components(&["a/b"]).unwrap(),
            GraphPath::from_components(&["a", "", "b"]).unwrap(),
        ];
        let mut encodings = std::collections::HashSet::new();
        for graph in paths {
            let record = WalRecord::lpg(
                transaction_id,
                graph.clone(),
                LpgMutationOp::CreateNode {
                    id: NodeId::new(19),
                    labels: vec!["ExactScope".into()],
                },
            );
            assert_eq!(record.transaction_id(), Some(transaction_id));
            assert!(record.is_lpg_mutation());
            assert!(record.is_data_mutation());
            let bytes =
                bincode::serde::encode_to_vec(&record, bincode::config::standard()).unwrap();
            let (decoded, consumed): (WalRecord, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(consumed, bytes.len());
            assert!(
                encodings.insert(bytes),
                "distinct literal paths must have distinct bytes"
            );
            for value in [roundtrip(&record), decoded] {
                assert!(
                    matches!(value, WalRecord::LpgMutation { graph: actual, .. } if actual == graph)
                );
            }
        }
    }

    #[test]
    fn exact_node_label_images_wire_preserves_order_and_empty_sets() {
        let images = vec![vec!["A".to_owned()], Vec::new(), Vec::new()];
        let record = WalRecord::lpg(
            TransactionId::new(71),
            GraphPath::from_components(&[""]).unwrap(),
            LpgMutationOp::NodeLabelImages {
                id: NodeId::new(19),
                birth: true,
                images: images.clone(),
            },
        );
        assert!(record.validate_recovery().is_ok());
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&record, config).unwrap();
        let (decoded, consumed): (WalRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(consumed, bytes.len());
        for value in [roundtrip(&record), decoded] {
            assert!(matches!(value, WalRecord::LpgMutation {
                transaction_id, graph,
                op: LpgMutationOp::NodeLabelImages { id, birth: true, images: actual },
            } if transaction_id == TransactionId::new(71) && graph.components() == [""]
                && id == NodeId::new(19) && actual == images));
        }
    }

    #[test]
    fn exact_node_label_images_wire_rejects_invalid_shapes() {
        for (id, images) in [
            (NodeId::INVALID, vec![Vec::new()]),
            (NodeId::new(1), Vec::new()),
            (NodeId::new(1), vec![vec!["A".into(), "A".into()]]),
            (
                NodeId::new(1),
                vec![(0..=u16::MAX).map(|label| label.to_string()).collect()],
            ),
        ] {
            let record = WalRecord::lpg(
                TransactionId::new(71),
                GraphPath::root(),
                LpgMutationOp::NodeLabelImages {
                    id,
                    birth: false,
                    images,
                },
            );
            assert!(record.validate_recovery().is_err());
        }
    }

    #[test]
    fn exact_node_label_images_reject_system_transaction() {
        let record = WalRecord::lpg(
            TransactionId::SYSTEM,
            GraphPath::root(),
            LpgMutationOp::NodeLabelImages {
                id: NodeId::new(1),
                birth: true,
                images: vec![vec!["A".into()]],
            },
        );
        assert!(record.validate_recovery().is_err());
    }

    #[test]
    fn test_graph_type_binding_roundtrip_and_tid() {
        let record = WalRecord::SetGraphTypeBinding {
            transaction_id: TransactionId::new(8),
            graph: GraphPath::from_components(&["analytics", "events"]).unwrap(),
            graph_type: Some("analytics/EventGraph".into()),
        };
        assert_eq!(record.transaction_id(), Some(TransactionId::new(8)));
        match roundtrip(&record) {
            WalRecord::SetGraphTypeBinding {
                transaction_id,
                graph,
                graph_type,
            } => {
                assert_eq!(transaction_id, TransactionId::new(8));
                assert_eq!(graph.components(), ["analytics", "events"]);
                assert_eq!(graph_type.as_deref(), Some("analytics/EventGraph"));
            }
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn named_graph_publication_wire_is_exact_and_transactional() {
        let transaction_id = TransactionId::new(7);
        let graph = GraphPath::from_components(&[""]).unwrap();
        let valid = WalRecord::lpg(transaction_id, graph.clone(), LpgMutationOp::PublishGraph);
        assert!(valid.validate_recovery().is_ok());
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&valid, config).unwrap();
        let (decoded, consumed): (WalRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(consumed, bytes.len());
        for record in [roundtrip(&valid), decoded] {
            assert!(matches!(record, WalRecord::LpgMutation {
                transaction_id: actual, graph: actual_graph, op: LpgMutationOp::PublishGraph,
            } if actual == transaction_id && actual_graph == graph));
        }
        for record in [
            WalRecord::lpg(
                transaction_id,
                GraphPath::root(),
                LpgMutationOp::PublishGraph,
            ),
            WalRecord::lpg(TransactionId::SYSTEM, graph, LpgMutationOp::PublishGraph),
        ] {
            assert!(record.validate_recovery().is_err());
        }
    }

    #[test]
    fn projection_v3_records_retain_exact_tags_and_reserved_tags_reject() {
        let declared_v3 = WalRecord::RdfLpgProjectionDeclaredV3 {
            projection_id: 17,
            mapping_digest: Digest256::from_bytes([0x44; 32]),
            mapping_format_version: 2,
            source_graph: Some("http://example.org/claims".into()),
            type_iri: "http://example.org/Person".into(),
            node_label: "Person".into(),
            epoch: EpochId::new(44),
        };
        let declared_bytes =
            bincode::serde::encode_to_vec(&declared_v3, bincode::config::standard()).unwrap();
        assert_eq!(
            declared_bytes,
            [
                48, 17, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
                68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 2, 1, 25, 104, 116, 116, 112,
                58, 47, 47, 101, 120, 97, 109, 112, 108, 101, 46, 111, 114, 103, 47, 99, 108, 97,
                105, 109, 115, 25, 104, 116, 116, 112, 58, 47, 47, 101, 120, 97, 109, 112, 108,
                101, 46, 111, 114, 103, 47, 80, 101, 114, 115, 111, 110, 6, 80, 101, 114, 115, 111,
                110, 44,
            ]
        );
        assert!(declared_v3.requires_sync());
        assert!(declared_v3.is_metadata());
        assert!(matches!(
            roundtrip(&declared_v3),
            WalRecord::RdfLpgProjectionDeclaredV3 {
                projection_id: 17,
                mapping_format_version: 2,
                source_graph: Some(graph),
                epoch,
                ..
            } if graph == "http://example.org/claims" && epoch == EpochId::new(44)
        ));

        let published_v3 = WalRecord::RdfLpgProjectionPublishedV3 {
            transaction_id: TransactionId::new(51),
            receipt: vec![0x47, 0x52, 0x50, 0x52, 3, 0],
        };
        assert_eq!(
            bincode::serde::encode_to_vec(&published_v3, bincode::config::standard()).unwrap(),
            [49, 51, 6, 71, 82, 80, 82, 3, 0]
        );
        assert!(!published_v3.requires_sync());
        assert!(!published_v3.is_metadata());
        assert!(published_v3.is_data_mutation());
        assert_eq!(published_v3.transaction_id(), Some(TransactionId::new(51)));
        assert!(matches!(
            roundtrip(&published_v3),
            WalRecord::RdfLpgProjectionPublishedV3 {
                transaction_id,
                receipt,
            } if transaction_id == TransactionId::new(51)
                && receipt == vec![0x47, 0x52, 0x50, 0x52, 3, 0]
        ));

        for reserved in [
            WalRecord::Reserved15,
            WalRecord::Reserved16,
            WalRecord::Reserved17,
            WalRecord::Reserved18,
            WalRecord::Reserved19,
            WalRecord::Reserved26,
            WalRecord::Reserved28,
            WalRecord::Reserved31,
            WalRecord::Reserved41,
            WalRecord::Reserved36,
            WalRecord::Reserved37,
            WalRecord::Reserved39,
            WalRecord::Reserved40,
        ] {
            assert!(reserved.validate_recovery().is_err());
            assert!(crate::wal::encode_record(&reserved).is_err());
            assert_eq!(reserved.transaction_id(), None);
            assert!(!reserved.is_data_mutation());
        }
    }

    #[test]
    fn recovery_validation_enforces_real_publication_epochs_without_losing_epoch_zero_history() {
        let committed_zero = WalRecord::Committed {
            transaction_id: TransactionId::new(7),
            epoch: EpochId::INITIAL,
        };
        assert!(committed_zero.validate_recovery().is_ok());
        assert!(
            WalRecord::EpochAdvance {
                epoch: EpochId::INITIAL
            }
            .validate_recovery()
            .is_ok()
        );
        let invalid = [
            WalRecord::Committed {
                transaction_id: TransactionId::new(7),
                epoch: EpochId::PENDING,
            },
            WalRecord::EpochAdvance {
                epoch: EpochId::PENDING,
            },
            WalRecord::CatalogBatchV3 {
                created_graph_incarnations: vec![],
                dropped_graph_incarnations: vec![],
                version: 2,
                epoch: EpochId::INITIAL,
                catalog_state: Vec::new(),
                created_graphs: Vec::new(),
                dropped_graphs: Vec::new(),
            },
            WalRecord::RdfLpgProjectionDeclaredV3 {
                projection_id: 17,
                mapping_digest: Digest256::from_bytes([0x44; Digest256::LEN]),
                mapping_format_version: 2,
                source_graph: None,
                type_iri: "http://example.org/Person".into(),
                node_label: "Person".into(),
                epoch: EpochId::INITIAL,
            },
        ];
        for record in invalid {
            assert!(
                record.validate_recovery().is_err(),
                "invalid recovery coordinate was accepted: {record:?}"
            );
        }

        let hidden_pending = WalRecord::CatalogBatchV2 {
            version: 1,
            records: vec![WalRecord::Committed {
                transaction_id: TransactionId::new(8),
                epoch: EpochId::PENDING,
            }],
        };
        assert!(hidden_pending.validate_recovery().is_err());
        let nested = WalRecord::CatalogBatchV2 {
            version: 1,
            records: vec![WalRecord::CatalogBatchV2 {
                version: 1,
                records: Vec::new(),
            }],
        };
        assert!(
            nested
                .validate_recovery()
                .expect_err("nested batches must fail without recursive descent")
                .contains("nested")
        );
    }

    #[test]
    fn recovery_validation_rejects_invalid_transaction_identity() {
        let record = WalRecord::Committed {
            transaction_id: TransactionId::INVALID,
            epoch: EpochId::new(1),
        };

        assert!(
            record
                .validate_recovery()
                .expect_err("INVALID must never authenticate a WAL transaction")
                .contains("TransactionId::INVALID")
        );
    }

    #[test]
    fn nested_catalog_batch_decode_fails_early_and_resets_its_guard() {
        let nested = WalRecord::CatalogBatchV2 {
            version: 1,
            records: vec![WalRecord::CatalogBatchV2 {
                version: 1,
                records: Vec::new(),
            }],
        };
        let nested_bytes =
            bincode::serde::encode_to_vec(&nested, bincode::config::standard()).unwrap();

        let error = bincode::serde::decode_from_slice::<WalRecord, _>(
            &nested_bytes,
            bincode::config::standard(),
        )
        .expect_err("nested catalog batches must fail before recursive materialization");
        assert!(
            error
                .to_string()
                .contains(NESTED_CATALOG_BATCH_DECODE_ERROR)
        );

        let valid = WalRecord::CatalogBatchV2 {
            version: 1,
            records: vec![WalRecord::CreateSchema { name: "s".into() }],
        };
        let valid_bytes =
            bincode::serde::encode_to_vec(&valid, bincode::config::standard()).unwrap();
        let (decoded, consumed) = bincode::serde::decode_from_slice::<WalRecord, _>(
            &valid_bytes,
            bincode::config::standard(),
        )
        .expect("a prior nested-batch error must not poison the decoder thread");
        assert_eq!(consumed, valid_bytes.len());
        assert!(matches!(
            decoded,
            WalRecord::CatalogBatchV2 { version: 1, records }
                if matches!(records.as_slice(), [WalRecord::CreateSchema { name }] if name == "s")
        ));
    }

    #[test]
    fn catalog_batch_v3_round_trips_and_forces_sync() {
        let record = WalRecord::CatalogBatchV3 {
            created_graph_incarnations: vec![
                grafeo_common::types::GraphIncarnationId::new(1),
                grafeo_common::types::GraphIncarnationId::new(2),
            ],
            dropped_graph_incarnations: vec![grafeo_common::types::GraphIncarnationId::new(3)],
            version: 2,
            epoch: EpochId::new(42),
            catalog_state: vec![0xCA, 0x7A, 0x10, 0x6],
            created_graphs: vec![
                grafeo_common::types::GraphPath::from_components(&["knowledge"]).unwrap(),
                grafeo_common::types::GraphPath::from_components(&["keep"]).unwrap(),
            ],
            dropped_graphs: vec![
                grafeo_common::types::GraphPath::from_components(&["stale"]).unwrap(),
            ],
        };

        assert!(record.requires_sync());
        assert!(record.is_metadata());
        assert!(record.transaction_id().is_none());

        match roundtrip(&record) {
            WalRecord::CatalogBatchV3 {
                version,
                epoch,
                catalog_state,
                created_graphs,
                dropped_graphs,
                created_graph_incarnations,
                dropped_graph_incarnations,
            } => {
                assert_eq!(
                    created_graph_incarnations
                        .iter()
                        .map(|id| id.as_u64())
                        .collect::<Vec<_>>(),
                    vec![1, 2]
                );
                assert_eq!(
                    dropped_graph_incarnations
                        .iter()
                        .map(|id| id.as_u64())
                        .collect::<Vec<_>>(),
                    vec![3]
                );
                assert_eq!(version, 2);
                assert_eq!(epoch, EpochId::new(42));
                assert_eq!(catalog_state, vec![0xCA, 0x7A, 0x10, 0x6]);
                assert_eq!(
                    created_graphs,
                    [
                        GraphPath::from_components(&["knowledge"]).unwrap(),
                        GraphPath::from_components(&["keep"]).unwrap()
                    ]
                );
                assert_eq!(
                    dropped_graphs,
                    [GraphPath::from_components(&["stale"]).unwrap()]
                );
            }
            other => panic!("Wrong variant: {other:?}"),
        }

        let encoded = bincode::serde::encode_to_vec(&record, bincode::config::standard()).unwrap();
        let (decoded, consumed): (WalRecord, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert!(matches!(
            decoded,
            WalRecord::CatalogBatchV3 {
                version: 2,
                epoch,
                catalog_state,
                ..
            } if epoch == EpochId::new(42)
                && catalog_state == vec![0xCA, 0x7A, 0x10, 0x6]
        ));
    }
}
