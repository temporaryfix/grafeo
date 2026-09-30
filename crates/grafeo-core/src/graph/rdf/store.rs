//! RDF Triple Store.
//!
//! Provides an in-memory triple store with efficient indexing for
//! subject, predicate, and object queries.

use super::sink::TripleSink;
use super::term::Term;
use super::triple::{Quad, Triple, TriplePattern};
use super::{RdfDatasetHistory, RdfGraphIdentity, RdfGraphLife, RdfHistoryError, RdfQuadVersion};
use grafeo_common::types::{
    EpochId, EpochInterval, GraphIncarnationId, HistoryCompleteness, StoreId,
    StoreIdGenerationError, TaiNanoseconds, TransactionId, ValidTimeInterval,
    WorldIdentityMetadataV1,
};
use grafeo_common::utils::hash::FxHashSet;
use hashbrown::HashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod replacement;
pub use replacement::{
    InstalledRdfDatasetReplacement, PreparedRdfDatasetReplacement, ReadyRdfDatasetReplacement,
};

// The primary key shares the retained triple instead of rendering three owned
// strings. Borrowed probes use the same canonical hash/equality without allocation.
#[derive(Clone)]
struct CanonicalTripleKey(Arc<Triple>);

struct CanonicalTripleRef<'a>([&'a Term; 3]);

impl<'a> From<&'a Triple> for CanonicalTripleRef<'a> {
    fn from(triple: &'a Triple) -> Self {
        Self([triple.subject(), triple.predicate(), triple.object()])
    }
}

impl Hash for CanonicalTripleRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for term in self.0 {
            if let Term::Literal(literal) = term
                && let Some(language) = literal.language()
            {
                std::mem::discriminant(term).hash(state);
                literal.value().hash(state);
                literal.datatype().hash(state);
                language.len().hash(state);
                for byte in language.bytes() {
                    byte.to_ascii_lowercase().hash(state);
                }
            } else {
                term.hash(state);
            }
        }
    }
}

impl Hash for CanonicalTripleKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        CanonicalTripleRef::from(self.0.as_ref()).hash(state);
    }
}

impl PartialEq for CanonicalTripleKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_identity(&other.0)
    }
}

impl Eq for CanonicalTripleKey {}

impl hashbrown::Equivalent<CanonicalTripleKey> for CanonicalTripleRef<'_> {
    fn equivalent(&self, key: &CanonicalTripleKey) -> bool {
        self.0
            .iter()
            .zip(CanonicalTripleRef::from(key.0.as_ref()).0)
            .all(|(left, right)| left.same_identity(right))
    }
}

/// Transaction-time (and optional valid-time) for one RDF quad version.
#[derive(Debug, Clone)]
pub struct QuadLife {
    /// Transaction-time interval `[from, to)`.
    pub tx: EpochInterval,
    /// Optional application valid-time on the canonical TAI nanosecond axis.
    pub valid: Option<ValidTimeInterval>,
}

/// A pending operation in a transaction buffer.
#[derive(Debug, Clone)]
enum PendingOp {
    /// Insert a triple.
    Insert {
        triple: Triple,
        valid: Option<ValidTimeInterval>,
    },
    /// Delete a triple.
    Delete(Triple),
}

/// Transaction buffer for pending operations.
#[derive(Default)]
struct TransactionBuffer {
    /// Pending operations for each transaction.
    buffers: HashMap<TransactionId, Vec<PendingOp>>,
    /// Committed epoch visible to snapshot-isolated transactions.
    ///
    /// Read-committed transactions are intentionally absent and continue to
    /// read the live indexes on every statement.
    snapshot_epochs: HashMap<TransactionId, EpochId>,
    /// Partition revision at the first snapshot or write; checked only for writers.
    write_revisions: HashMap<TransactionId, u64>,
    /// Detached named-graph partitions created by each transaction.
    created_graphs: HashMap<TransactionId, HashMap<String, Arc<RdfStore>>>,
    /// Exact shared partitions staged for removal, with their observed revision.
    dropped_graphs: HashMap<TransactionId, HashMap<String, RdfGraphPin>>,
    /// Existing shared partitions written by each transaction.
    ///
    /// Commit validates both the exact `Arc` and revision, preventing a
    /// concurrent DROP from orphaning writes and preventing DROP from erasing a
    /// graph that changed after it was pinned.
    touched_graphs: HashMap<TransactionId, HashMap<String, RdfGraphPin>>,
    /// Exact committed named-graph partitions (or absence) observed by a
    /// snapshot-isolated transaction.
    ///
    /// These are deliberately separate from `touched_graphs`: a pure read
    /// must keep resolving the same graph incarnation after a concurrent
    /// DROP/CREATE, but it must not acquire write-conflict semantics.
    read_graphs: HashMap<TransactionId, HashMap<String, Option<Arc<RdfStore>>>>,
    /// Transactions that have enumerated the complete named-graph catalog.
    ///
    /// Once the catalog has been observed, an otherwise-unpinned name is
    /// pinned absent so a later CREATE cannot appear as a phantom.
    snapshotted_graph_catalogs: FxHashSet<TransactionId>,
}

#[derive(Clone)]
struct RdfGraphPin {
    store: Arc<RdfStore>,
    revision: u64,
}

/// One retained named-graph partition and its durable lifecycle interval.
#[derive(Clone)]
struct NamedGraphHistory {
    name: String,
    incarnation: GraphIncarnationId,
    tx: EpochInterval,
    store: Arc<RdfStore>,
}

/// Opaque snapshot of one transaction's pending RDF operations.
///
/// This covers the default graph, every live named-graph partition, and every
/// exact partition retained by this transaction at capture time. Detached graph
/// lifecycle maps and exact shared-graph pins are included, so
/// rollback-to-savepoint can rewind graph DDL without exposing an uncommitted
/// partition to another session.
#[derive(Clone)]
pub struct RdfTransactionSavepoint {
    default_ops: Vec<PendingOp>,
    default_snapshot_epoch: Option<EpochId>,
    default_write_revision: Option<u64>,
    created_graphs: HashMap<String, Arc<RdfStore>>,
    dropped_graphs: HashMap<String, RdfGraphPin>,
    touched_graphs: HashMap<String, RdfGraphPin>,
    named: Vec<RdfNamedTransactionSavepoint>,
}

#[derive(Clone)]
struct RdfNamedTransactionSavepoint {
    store: Arc<RdfStore>,
    ops: Vec<PendingOp>,
    snapshot_epoch: Option<EpochId>,
    write_revision: Option<u64>,
    created_graphs: HashMap<String, Arc<RdfStore>>,
    dropped_graphs: HashMap<String, RdfGraphPin>,
    touched_graphs: HashMap<String, RdfGraphPin>,
}

/// Configuration for the RDF store.
#[derive(Debug, Clone)]
pub struct RdfStoreConfig {
    /// Initial capacity for triple storage.
    pub initial_capacity: usize,
    /// Whether to build object index (for reverse lookups).
    pub index_objects: bool,
}

impl Default for RdfStoreConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            index_objects: true,
        }
    }
}

/// Store-scoped proof that this thread holds one RDF store's commit gate.
///
/// The private fields make this proof unforgeable outside this module. A
/// guard from one store cannot authorize an under-gate operation on another.
#[doc(hidden)]
pub struct RdfCommitGuard<'a> {
    store: &'a RdfStore,
    _guard: parking_lot::MutexGuard<'a, ()>,
}

fn tx_within_truthful_boundary(
    tx: EpochInterval,
    completeness: HistoryCompleteness,
) -> Option<EpochInterval> {
    let Some(boundary) = completeness.authoritative_from() else {
        return Some(tx);
    };
    if !tx.is_open() && tx.to() <= boundary {
        return None;
    }
    let from = tx.from().max(boundary);
    Some(if tx.is_open() {
        EpochInterval::open(from)
    } else {
        EpochInterval::closed(from, tx.to())
    })
}

fn history_life_within_truthful_boundary(
    mut life: QuadLife,
    completeness: HistoryCompleteness,
) -> Option<QuadLife> {
    life.tx = tx_within_truthful_boundary(life.tx, completeness)?;
    Some(life)
}

/// An in-memory RDF triple store.
///
/// The store maintains multiple indexes for efficient querying:
/// - SPO (Subject, Predicate, Object): primary storage
/// - POS (Predicate, Object, Subject): for predicate-based queries
/// - OSP (Object, Subject, Predicate): for object-based queries (optional)
///
/// The store also supports transactional operations through buffering.
/// When operations are performed within a transaction context, they are
/// buffered until commit (applied) or rollback (discarded).
pub struct RdfStore {
    /// Configuration.
    config: RdfStoreConfig,
    /// Canonical membership and its first lossless live representative.
    triples: RwLock<HashMap<CanonicalTripleKey, Arc<Triple>>>,
    /// Subject index: subject -> triples.
    subject_index: RwLock<hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Predicate index: predicate -> triples.
    predicate_index:
        RwLock<hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Object index: object -> triples (optional).
    object_index:
        RwLock<Option<hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>>>,
    /// Subject+Predicate composite index: (subject, predicate) -> triples.
    sp_index:
        RwLock<hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Predicate+Object composite index: (predicate, object) -> triples.
    po_index:
        RwLock<hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Object+Subject composite index: (object, subject) -> triples.
    os_index:
        RwLock<hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Transaction buffers for pending operations.
    tx_buffer: RwLock<TransactionBuffer>,
    /// Named graphs, each a separate `RdfStore` partition.
    named_graphs: RwLock<HashMap<String, Arc<RdfStore>>>,
    /// Cached RDF statistics for query optimization. Invalidated on any mutation.
    statistics_cache: RwLock<Option<Arc<crate::statistics::RdfStatistics>>>,
    /// Cached term dictionary for dictionary-encoded triple scans. Invalidated on any mutation.
    dictionary_cache: RwLock<Option<Arc<super::dictionary::TermDictionary>>>,
    /// Compact Ring Index for SPARQL query acceleration. Built during `bulk_load()`
    /// or explicitly via `rebuild_ring()`. Marked stale on incremental mutations,
    /// falling back to HashMap indexes until rebuilt.
    #[cfg(feature = "ring-index")]
    ring: RwLock<Option<Arc<crate::index::ring::TripleRing>>>,
    /// Whether the Ring is stale (triples changed since last build).
    #[cfg(feature = "ring-index")]
    ring_stale: std::sync::atomic::AtomicBool,
    /// Serializes epoch assignment with applying buffered RDF ops.
    commit_lock: Mutex<()>,
    /// Last committed epoch used to stamp RDF tx-time.
    commit_epoch: AtomicU64,
    /// Quad versions: live + historical intervals.
    history: RwLock<HashMap<Arc<Triple>, Vec<QuadLife>>>,
    /// Portable identity shared by every RDF partition in this dataset.
    store_id: RwLock<StoreId>,
    /// Whether `store_id` was verified from a current durable artifact.
    ///
    /// Fresh startup identities and identities synthesized while importing a
    /// legacy format remain provisional until WAL metadata confirms them.
    durable_identity: AtomicBool,
    /// Identity of this partition (`0` for the dataset default graph).
    graph_incarnation: GraphIncarnationId,
    /// Dataset-wide monotonic named-graph incarnation allocator.
    next_graph_incarnation: Arc<AtomicU64>,
    /// Incarnations observed or reserved in this process, including
    /// zero-length and aborted graph lifetimes.
    reserved_graph_incarnations: Arc<RwLock<FxHashSet<GraphIncarnationId>>>,
    /// Current and dropped named-graph partitions with exact lifetimes.
    named_graph_history: RwLock<Vec<NamedGraphHistory>>,
    /// Truthful boundary for imported legacy history.
    history_completeness: RwLock<HistoryCompleteness>,
    /// Serializes lifecycle registry updates with dataset-history snapshots.
    history_lifecycle_lock: Mutex<()>,
    /// Declared RDF→LPG projections (`id` → type_iri, label, rebuilt_at).
    projections: RwLock<HashMap<u64, (String, String, Option<EpochId>)>>,
    /// Database authority scope required by public mutators (`0` = unsealed).
    mutation_scope: AtomicU64,
    /// Monotonic committed-content revision used by named-graph CAS validation.
    lifecycle_revision: AtomicU64,
}

impl RdfStore {
    /// Creates a new RDF store with default configuration.
    ///
    /// # Panics
    ///
    /// Panics if the operating system cannot provide cryptographic entropy.
    /// Persistence-aware callers should generate the database identity once
    /// and use [`Self::with_config_and_store_id`] instead.
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(|error| {
            panic!("failed to generate RDF store identity from system entropy: {error}")
        })
    }

    /// Creates a new RDF store with a fallible system-generated identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system cannot provide cryptographic
    /// entropy.
    pub fn try_new() -> Result<Self, StoreIdGenerationError> {
        Self::try_with_config(RdfStoreConfig::default())
    }

    /// Creates a new RDF store with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if the operating system cannot provide cryptographic entropy.
    /// Persistence-aware callers should use
    /// [`Self::with_config_and_store_id`].
    pub fn with_config(config: RdfStoreConfig) -> Self {
        Self::try_with_config(config).unwrap_or_else(|error| {
            panic!("failed to generate RDF store identity from system entropy: {error}")
        })
    }

    /// Creates a new RDF store with the given configuration and a fallible
    /// system-generated identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system cannot provide cryptographic
    /// entropy.
    pub fn try_with_config(config: RdfStoreConfig) -> Result<Self, StoreIdGenerationError> {
        let store_id = StoreId::generate()?;
        Ok(Self::with_config_and_store_id(config, store_id))
    }

    /// Creates the root RDF dataset with an engine-owned logical identity.
    ///
    /// Named graph partitions inherit this exact value. Persistence-aware
    /// engines use this constructor so restore reuses the identity carried by
    /// the durable artifact instead of allocating a process-local namespace.
    #[must_use]
    pub fn with_config_and_store_id(config: RdfStoreConfig, store_id: StoreId) -> Self {
        Self::with_dataset_identity(
            config,
            store_id,
            GraphIncarnationId::DEFAULT_GRAPH,
            Arc::new(AtomicU64::new(GraphIncarnationId::FIRST_NAMED.as_u64())),
            Arc::new(RwLock::new(FxHashSet::default())),
        )
    }

    fn with_dataset_identity(
        config: RdfStoreConfig,
        store_id: StoreId,
        graph_incarnation: GraphIncarnationId,
        next_graph_incarnation: Arc<AtomicU64>,
        reserved_graph_incarnations: Arc<RwLock<FxHashSet<GraphIncarnationId>>>,
    ) -> Self {
        let object_index = if config.index_objects {
            Some(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            ))
        } else {
            None
        };

        Self {
            triples: RwLock::new(HashMap::new()),
            subject_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            predicate_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            object_index: RwLock::new(object_index),
            sp_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            po_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            os_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            tx_buffer: RwLock::new(TransactionBuffer::default()),
            named_graphs: RwLock::new(HashMap::new()),
            statistics_cache: RwLock::new(None),
            dictionary_cache: RwLock::new(None),
            #[cfg(feature = "ring-index")]
            ring: RwLock::new(None),
            #[cfg(feature = "ring-index")]
            ring_stale: std::sync::atomic::AtomicBool::new(false),
            commit_lock: Mutex::new(()),
            commit_epoch: AtomicU64::new(0),
            history: RwLock::new(HashMap::new()),
            store_id: RwLock::new(store_id),
            durable_identity: AtomicBool::new(false),
            graph_incarnation,
            next_graph_incarnation,
            reserved_graph_incarnations,
            named_graph_history: RwLock::new(Vec::new()),
            history_completeness: RwLock::new(HistoryCompleteness::Complete),
            history_lifecycle_lock: Mutex::new(()),
            projections: RwLock::new(HashMap::new()),
            mutation_scope: AtomicU64::new(0),
            lifecycle_revision: AtomicU64::new(0),
            config,
        }
    }

    /// Seals this store to one database-scoped mutation authority.
    ///
    /// Returns `false` if the store was already sealed by a different
    /// authority. Named substores inherit the same scope recursively.
    pub fn seal_unframed_writes(
        &self,
        authority: &crate::graph::write_permit::WriteAuthority,
    ) -> bool {
        self.seal_with_scope(authority.scope().get())
    }

    fn seal_with_scope(&self, scope: u64) -> bool {
        match self.mutation_scope.compare_exchange(
            0,
            scope,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(existing) if existing == scope => {}
            Err(_) => return false,
        }
        for g in self.named_graphs.read().values() {
            if !g.seal_with_scope(scope) {
                return false;
            }
        }
        for graphs in self.tx_buffer.read().created_graphs.values() {
            for graph in graphs.values() {
                if !graph.seal_with_scope(scope) {
                    return false;
                }
            }
        }
        true
    }

    fn unframed_writes_allowed(&self) -> bool {
        let scope = self
            .mutation_scope
            .load(std::sync::atomic::Ordering::Acquire);
        scope == 0
            || std::num::NonZeroU64::new(scope).is_some_and(crate::graph::write_permit::is_held)
    }

    fn retain_exact_partition(
        partitions: &mut Vec<Arc<RdfStore>>,
        seen: &mut FxHashSet<*const RdfStore>,
        partition: &Arc<RdfStore>,
    ) {
        if seen.insert(Arc::as_ptr(partition)) {
            partitions.push(Arc::clone(partition));
        }
    }

    /// Captures pending RDF operations for `transaction_id` across the dataset.
    ///
    /// The returned token pins exact named-graph partitions, so a later restore
    /// never redirects buffered operations through a reused graph name.
    #[must_use]
    pub fn transaction_savepoint(&self, transaction_id: TransactionId) -> RdfTransactionSavepoint {
        let default = self.tx_buffer.read();
        let default_ops = default
            .buffers
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default();
        let default_snapshot_epoch = default.snapshot_epochs.get(&transaction_id).copied();
        let default_write_revision = default.write_revisions.get(&transaction_id).copied();
        let created_graphs = default
            .created_graphs
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default();
        let dropped_graphs = default
            .dropped_graphs
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default();
        let touched_graphs = default
            .touched_graphs
            .get(&transaction_id)
            .cloned()
            .unwrap_or_default();
        drop(default);

        let mut named_stores = Vec::new();
        let mut seen = FxHashSet::default();
        for store in self.named_graphs.read().values() {
            Self::retain_exact_partition(&mut named_stores, &mut seen, store);
        }
        for store in created_graphs.values() {
            Self::retain_exact_partition(&mut named_stores, &mut seen, store);
        }
        for pin in dropped_graphs.values() {
            Self::retain_exact_partition(&mut named_stores, &mut seen, &pin.store);
        }
        for pin in touched_graphs.values() {
            Self::retain_exact_partition(&mut named_stores, &mut seen, &pin.store);
        }
        let named = named_stores
            .into_iter()
            .map(|store| {
                let buffer = store.tx_buffer.read();
                let ops = buffer
                    .buffers
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default();
                let snapshot_epoch = buffer.snapshot_epochs.get(&transaction_id).copied();
                let write_revision = buffer.write_revisions.get(&transaction_id).copied();
                let created_graphs = buffer
                    .created_graphs
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default();
                let dropped_graphs = buffer
                    .dropped_graphs
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default();
                let touched_graphs = buffer
                    .touched_graphs
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default();
                drop(buffer);
                RdfNamedTransactionSavepoint {
                    store,
                    ops,
                    snapshot_epoch,
                    write_revision,
                    created_graphs,
                    dropped_graphs,
                    touched_graphs,
                }
            })
            .collect();

        RdfTransactionSavepoint {
            default_ops,
            default_snapshot_epoch,
            default_write_revision,
            created_graphs,
            dropped_graphs,
            touched_graphs,
            named,
        }
    }

    /// Restores pending RDF operations to a prior transaction savepoint.
    ///
    /// Partitions first touched after the savepoint have only this
    /// transaction's buffer removed; other transactions' buffers and all
    /// committed triples remain untouched.
    pub fn restore_transaction_savepoint(
        &self,
        transaction_id: TransactionId,
        snapshot: &RdfTransactionSavepoint,
    ) {
        if !self.unframed_writes_allowed() {
            return;
        }

        let mut current = Vec::new();
        let mut seen = FxHashSet::default();
        for store in self.named_graphs.read().values() {
            Self::retain_exact_partition(&mut current, &mut seen, store);
        }
        {
            let mut buffer = self.tx_buffer.write();
            if let Some(created) = buffer.created_graphs.get(&transaction_id) {
                for store in created.values() {
                    Self::retain_exact_partition(&mut current, &mut seen, store);
                }
            }
            if let Some(dropped) = buffer.dropped_graphs.get(&transaction_id) {
                for pin in dropped.values() {
                    Self::retain_exact_partition(&mut current, &mut seen, &pin.store);
                }
            }
            if let Some(touched) = buffer.touched_graphs.get(&transaction_id) {
                for pin in touched.values() {
                    Self::retain_exact_partition(&mut current, &mut seen, &pin.store);
                }
            }
            if snapshot.default_ops.is_empty() {
                buffer.buffers.remove(&transaction_id);
            } else {
                buffer
                    .buffers
                    .insert(transaction_id, snapshot.default_ops.clone());
            }
            match snapshot.default_snapshot_epoch {
                Some(epoch) => {
                    buffer.snapshot_epochs.insert(transaction_id, epoch);
                }
                None => {
                    buffer.snapshot_epochs.remove(&transaction_id);
                }
            }
            match snapshot.default_write_revision {
                Some(revision) => {
                    buffer.write_revisions.insert(transaction_id, revision);
                }
                None => {
                    buffer.write_revisions.remove(&transaction_id);
                }
            }
            if snapshot.created_graphs.is_empty() {
                buffer.created_graphs.remove(&transaction_id);
            } else {
                buffer
                    .created_graphs
                    .insert(transaction_id, snapshot.created_graphs.clone());
            }
            if snapshot.dropped_graphs.is_empty() {
                buffer.dropped_graphs.remove(&transaction_id);
            } else {
                buffer
                    .dropped_graphs
                    .insert(transaction_id, snapshot.dropped_graphs.clone());
            }
            if snapshot.touched_graphs.is_empty() {
                buffer.touched_graphs.remove(&transaction_id);
            } else {
                buffer
                    .touched_graphs
                    .insert(transaction_id, snapshot.touched_graphs.clone());
            }
        }

        let saved_partitions: FxHashSet<*const RdfStore> = snapshot
            .named
            .iter()
            .map(|saved| Arc::as_ptr(&saved.store))
            .collect();
        for store in current {
            if !saved_partitions.contains(&Arc::as_ptr(&store)) {
                let mut buffer = store.tx_buffer.write();
                buffer.buffers.remove(&transaction_id);
                buffer.snapshot_epochs.remove(&transaction_id);
                buffer.write_revisions.remove(&transaction_id);
                buffer.created_graphs.remove(&transaction_id);
                buffer.dropped_graphs.remove(&transaction_id);
                buffer.touched_graphs.remove(&transaction_id);
            }
        }
        for saved in &snapshot.named {
            let mut buffer = saved.store.tx_buffer.write();
            if saved.ops.is_empty() {
                buffer.buffers.remove(&transaction_id);
            } else {
                buffer.buffers.insert(transaction_id, saved.ops.clone());
            }
            match saved.snapshot_epoch {
                Some(epoch) => {
                    buffer.snapshot_epochs.insert(transaction_id, epoch);
                }
                None => {
                    buffer.snapshot_epochs.remove(&transaction_id);
                }
            }
            match saved.write_revision {
                Some(revision) => {
                    buffer.write_revisions.insert(transaction_id, revision);
                }
                None => {
                    buffer.write_revisions.remove(&transaction_id);
                }
            }
            if saved.created_graphs.is_empty() {
                buffer.created_graphs.remove(&transaction_id);
            } else {
                buffer
                    .created_graphs
                    .insert(transaction_id, saved.created_graphs.clone());
            }
            if saved.dropped_graphs.is_empty() {
                buffer.dropped_graphs.remove(&transaction_id);
            } else {
                buffer
                    .dropped_graphs
                    .insert(transaction_id, saved.dropped_graphs.clone());
            }
            if saved.touched_graphs.is_empty() {
                buffer.touched_graphs.remove(&transaction_id);
            } else {
                buffer
                    .touched_graphs
                    .insert(transaction_id, saved.touched_graphs.clone());
            }
        }
    }

    /// Records a declared RDF→LPG projection.
    pub fn remember_projection(&self, id: u64, type_iri: &str, node_label: &str) {
        if !self.unframed_writes_allowed() {
            return;
        }
        self.projections
            .write()
            .entry(id)
            .or_insert_with(|| (type_iri.to_string(), node_label.to_string(), None));
    }

    /// Projection spec: (type_iri, node_label).
    #[must_use]
    pub fn projection(&self, id: u64) -> Option<(u64, String, String)> {
        self.projections
            .read()
            .get(&id)
            .map(|(t, l, _)| (id, t.clone(), l.clone()))
    }

    /// Records last rebuild epoch for lag.
    pub fn mark_projection_rebuilt(&self, id: u64, epoch: EpochId) {
        if !self.unframed_writes_allowed() {
            return;
        }
        if let Some(ent) = self.projections.write().get_mut(&id) {
            ent.2 = Some(epoch);
        }
    }

    /// Epoch of last rebuild.
    #[must_use]
    pub fn projection_rebuilt_at(&self, id: u64) -> Option<EpochId> {
        self.projections.read().get(&id).and_then(|e| e.2)
    }

    /// Sets the real epoch used to stamp subsequent insert/delete tx-time.
    ///
    /// Never moves the stamp backwards: concurrent applies keep the max.
    /// Epoch zero is a valid empty-store/recovery frontier. The pending epoch
    /// sentinel is reserved for open intervals and is never a commit stamp.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for [`EpochId::PENDING`] or an invalid
    /// transaction-state error when this thread lacks the authority for a
    /// sealed store. The default graph, named graphs, and transaction-detached
    /// graphs are all left unchanged on error.
    pub fn try_set_commit_epoch(&self, epoch: EpochId) -> grafeo_common::utils::error::Result<()> {
        use grafeo_common::utils::error::{Error, TransactionError};

        if epoch == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "RDF commit epoch cannot be PENDING".to_string(),
            ));
        }
        if !self.unframed_writes_allowed() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF commit-epoch update lacks mutation authority".to_string(),
            )));
        }
        self.commit_epoch.fetch_max(epoch.0, Ordering::SeqCst);
        let graphs: Vec<Arc<RdfStore>> = self.named_graphs.read().values().cloned().collect();
        for g in graphs {
            g.commit_epoch.fetch_max(epoch.0, Ordering::SeqCst);
        }
        let detached: Vec<Arc<RdfStore>> = self
            .tx_buffer
            .read()
            .created_graphs
            .values()
            .flat_map(|graphs| graphs.values().cloned())
            .collect();
        for graph in detached {
            graph.commit_epoch.fetch_max(epoch.0, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Holds the RDF commit gate so epoch assignment and apply stay ordered.
    pub fn lock_commit(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.commit_lock.lock()
    }

    /// Attempts to hold the RDF commit gate without waiting for its owner.
    pub fn try_lock_commit(&self) -> Option<parking_lot::MutexGuard<'_, ()>> {
        self.commit_lock.try_lock()
    }

    /// Holds the RDF commit gate and returns a store-scoped proof for recovery
    /// operations that must preserve an outer database lock order.
    #[doc(hidden)]
    pub fn lock_commit_scoped(&self) -> RdfCommitGuard<'_> {
        RdfCommitGuard {
            store: self,
            _guard: self.commit_lock.lock(),
        }
    }

    /// True if any quad versions (live or historical) exist.
    #[must_use]
    pub fn has_history(&self) -> bool {
        !self.history.read().is_empty()
    }

    /// Epoch used to stamp RDF transaction-time.
    #[must_use]
    pub fn commit_epoch(&self) -> EpochId {
        EpochId::new(self.commit_epoch.load(Ordering::Relaxed))
    }

    /// Portable identity of the logical RDF store.
    #[must_use]
    pub fn store_id(&self) -> StoreId {
        *self.store_id.read()
    }

    /// Identity of this RDF graph partition within the dataset.
    #[must_use]
    pub const fn graph_incarnation(&self) -> GraphIncarnationId {
        self.graph_incarnation
    }

    /// First named-graph incarnation not yet reserved by this dataset.
    #[must_use]
    pub fn next_graph_incarnation(&self) -> GraphIncarnationId {
        GraphIncarnationId::new(self.next_graph_incarnation.load(Ordering::Acquire))
    }

    /// Applies a durable allocator high-water mark during recovery.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign store, the reserved default-graph value,
    /// or missing recovery mutation authority.
    #[doc(hidden)]
    pub fn adopt_graph_incarnation_high_water(
        &self,
        store_id: StoreId,
        next: GraphIncarnationId,
    ) -> Result<(), String> {
        if !self.unframed_writes_allowed() {
            return Err("RDF graph-incarnation recovery lacks mutation authority".to_string());
        }
        // Exact replacement holds this same writer while changing the dataset
        // identity and allocator cut. Validate provenance only after admission.
        let _reserved = self.reserved_graph_incarnations.write();
        let current_store_id = self.store_id();
        if store_id != current_store_id {
            return Err(format!(
                "RDF graph-incarnation metadata belongs to store {store_id}, not {}",
                current_store_id
            ));
        }
        if next.is_default_graph() {
            return Err("RDF graph-incarnation high-water must be non-zero".to_string());
        }
        self.next_graph_incarnation
            .fetch_max(next.as_u64(), Ordering::AcqRel);
        Ok(())
    }

    /// Truthful provenance boundary of the persisted temporal history.
    #[must_use]
    pub fn history_completeness(&self) -> HistoryCompleteness {
        *self.history_completeness.read()
    }

    /// Adopts durable identity metadata before WAL data replay.
    ///
    /// An already-populated dataset may only confirm the exact identity it
    /// already carries. An empty startup store may adopt the persisted value
    /// before any graph or statement handle is reconstructed.
    ///
    /// # Errors
    ///
    /// Returns an error without changing the store when mutation authority is
    /// absent or populated state conflicts with the durable identity.
    #[doc(hidden)]
    pub fn adopt_recovery_identity(
        &self,
        identity: &WorldIdentityMetadataV1,
    ) -> Result<(), String> {
        if !self.unframed_writes_allowed() {
            return Err("RDF recovery identity adoption lacks mutation authority".to_string());
        }
        let _commit = self.commit_lock.lock();
        let _lifecycle = self.history_lifecycle_lock.lock();
        let current = WorldIdentityMetadataV1::new(self.store_id(), self.history_completeness())
            .map_err(|error| format!("invalid current RDF identity metadata: {error}"))?;
        if current == *identity {
            self.durable_identity.store(true, Ordering::Release);
            return Ok(());
        }
        if self.durable_identity.load(Ordering::Acquire) {
            return Err(format!(
                "durable RDF identity conflicts with recovered metadata: current {:?}, recovered {:?}",
                current, identity
            ));
        }
        let populated = !self.triples.read().is_empty()
            || !self.history.read().is_empty()
            || !self.named_graphs.read().is_empty()
            || !self.named_graph_history.read().is_empty();
        if populated {
            return Err(format!(
                "durable RDF identity conflicts with populated store: current {:?}, recovered {:?}",
                current, identity
            ));
        }
        *self.store_id.write() = identity.store_id();
        *self.history_completeness.write() = identity.history();
        self.durable_identity.store(true, Ordering::Release);
        self.lifecycle_revision.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Captures authoritative graph-qualified RDF interval history.
    ///
    /// Dropped graph partitions remain included. Consumers derive cuts, ordered
    /// diffs, and durable CDC pages from this one persisted representation.
    ///
    /// # Errors
    ///
    /// Returns an error if any in-memory lifecycle, interval, or statement
    /// identity invariant is inconsistent.
    pub fn dataset_history(&self) -> Result<RdfDatasetHistory, RdfHistoryError> {
        let _commit = self.commit_lock.lock();
        self.dataset_history_under_commit_gate()
    }

    /// Captures dataset history while the caller already holds [`Self::lock_commit`].
    ///
    /// Persistence/checkpoint paths use this to avoid recursively acquiring the
    /// non-reentrant commit gate. Ordinary callers must use [`Self::dataset_history`].
    #[doc(hidden)]
    pub fn dataset_history_under_commit_gate(&self) -> Result<RdfDatasetHistory, RdfHistoryError> {
        let _lifecycle = self.history_lifecycle_lock.lock();
        let store_id = self.store_id();
        let completeness = self.history_completeness();
        let mut versions = Vec::new();
        for (triple, lives) in self.quad_history() {
            for life in lives {
                let Some(life) = history_life_within_truthful_boundary(life, completeness) else {
                    continue;
                };
                versions.push(RdfQuadVersion::new(
                    store_id,
                    Quad::new((*triple).clone()),
                    GraphIncarnationId::DEFAULT_GRAPH,
                    life.tx,
                    life.valid,
                )?);
            }
        }

        let archives = self.named_graph_history.read().clone();
        let mut graph_lives = Vec::with_capacity(archives.len());
        for archive in archives {
            let graph = RdfGraphIdentity::named(archive.name.clone(), archive.incarnation)?;
            let Some(graph_tx) = tx_within_truthful_boundary(archive.tx, completeness) else {
                continue;
            };
            graph_lives.push(RdfGraphLife::new(graph, graph_tx)?);
            for (triple, lives) in archive.store.quad_history() {
                for life in lives {
                    let Some(life) = history_life_within_truthful_boundary(life, completeness)
                    else {
                        continue;
                    };
                    versions.push(RdfQuadVersion::new(
                        store_id,
                        Quad::named((*triple).clone(), archive.name.clone()),
                        archive.incarnation,
                        life.tx,
                        life.valid,
                    )?);
                }
            }
        }
        versions.sort_by_key(|version| (version.statement(), version.tx().from()));
        graph_lives.sort_by_key(|life| (life.graph().clone(), life.tx().from()));
        RdfDatasetHistory::new_with_high_water(
            store_id,
            completeness,
            GraphIncarnationId::new(self.next_graph_incarnation.load(Ordering::Acquire)),
            graph_lives,
            versions,
        )
    }

    fn current_commit_epoch(&self) -> EpochId {
        self.commit_epoch()
    }

    fn stamp_insert(&self, triple: &Arc<Triple>, epoch: EpochId, valid: Option<ValidTimeInterval>) {
        let mut history = self.history.write();
        let versions = history.entry(Arc::clone(triple)).or_default();
        if versions.last().is_some_and(|v| v.tx.is_open()) {
            return;
        }
        versions.push(QuadLife {
            tx: EpochInterval::open(epoch),
            valid,
        });
    }

    fn stamp_delete(&self, triple: &Triple, epoch: EpochId) {
        let mut history = self.history.write();
        let mut remove_entry = false;
        if let Some(versions) = history.get_mut(triple) {
            if let Some(last) = versions.last_mut()
                && last.tx.is_open()
            {
                if last.tx.from() == epoch {
                    versions.pop();
                } else {
                    last.tx = EpochInterval::closed(last.tx.from(), epoch);
                }
            }
            remove_entry = versions.is_empty();
        }
        if remove_entry {
            history.remove(triple);
        }
    }

    /// Inserts a triple with a legacy microsecond valid-time interval.
    ///
    /// New code should use [`insert_with_valid_time`](Self::insert_with_valid_time).
    /// An empty or inverted interval is rejected without mutating the store.
    pub fn insert_with_valid(&self, triple: Triple, valid_from: i64, valid_to: i64) -> bool {
        let Ok(valid) = ValidTimeInterval::from_legacy_micros(valid_from, valid_to) else {
            return false;
        };
        self.insert_with_valid_time(triple, valid)
    }

    /// Inserts a triple with canonical TAI-nanosecond application valid-time.
    pub fn insert_with_valid_time(&self, triple: Triple, valid: ValidTimeInterval) -> bool {
        self.insert_at_current_epoch_with_valid(triple, Some(valid))
    }

    /// Triples whose transaction-time interval contains `epoch`.
    #[must_use]
    pub fn triples_at_epoch(&self, epoch: EpochId) -> Vec<Arc<Triple>> {
        self.history
            .read()
            .iter()
            .filter(|(_, versions)| versions.iter().any(|v| v.tx.contains(epoch)))
            .map(|(t, _)| Arc::clone(t))
            .collect()
    }

    fn find_at_epoch(&self, pattern: &TriplePattern, epoch: EpochId) -> Vec<Arc<Triple>> {
        self.history
            .read()
            .iter()
            .filter(|(triple, versions)| {
                pattern.matches(triple) && versions.iter().any(|version| version.tx.contains(epoch))
            })
            .map(|(triple, _)| Arc::clone(triple))
            .collect()
    }

    /// Triples whose optional valid-time contains `at` (always-valid if none).
    #[must_use]
    pub fn triples_at_valid_time(&self, at: TaiNanoseconds) -> Vec<Arc<Triple>> {
        self.history
            .read()
            .iter()
            .filter(|(_, versions)| {
                versions.iter().any(|v| match v.valid {
                    None => v.tx.is_open(),
                    Some(valid) => valid.contains(at) && v.tx.is_open(),
                })
            })
            .map(|(t, _)| Arc::clone(t))
            .collect()
    }

    /// Compatibility query using the legacy microsecond coordinate.
    #[must_use]
    pub fn triples_at_valid(&self, at_micros: i64) -> Vec<Arc<Triple>> {
        self.triples_at_valid_time(TaiNanoseconds::from_legacy_micros(at_micros))
    }

    /// Added and removed triples between `from` (exclusive of removals at `from`) and `to`.
    #[must_use]
    pub fn diff_epochs(&self, from: EpochId, to: EpochId) -> (Vec<Arc<Triple>>, Vec<Arc<Triple>>) {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        for (triple, versions) in self.history.read().iter() {
            let at_from = versions.iter().any(|v| v.tx.contains(from));
            let at_to = versions.iter().any(|v| v.tx.contains(to));
            if !at_from && at_to {
                added.push(Arc::clone(triple));
            } else if at_from && !at_to {
                removed.push(Arc::clone(triple));
            }
        }
        (added, removed)
    }

    /// All quad versions (for persist).
    #[must_use]
    pub fn quad_history(&self) -> Vec<(Arc<Triple>, Vec<QuadLife>)> {
        self.history
            .read()
            .iter()
            .map(|(t, v)| (Arc::clone(t), v.clone()))
            .collect()
    }

    /// Restores a historical quad version (snapshot load). Does not touch live indexes
    /// unless the interval is still open.
    pub fn restore_quad_version(&self, triple: Triple, life: QuadLife) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let live = life.tx.is_open();
        {
            let mut history = self.history.write();
            history
                .entry(Arc::new(triple.clone()))
                .or_default()
                .push(life);
        }
        if live {
            let _ = self.insert(triple);
        }
    }

    /// Replaces this store from the authoritative graph-qualified history model.
    ///
    /// This preserves store identity, dropped graph partitions, graph
    /// incarnations, statement handles, and the exact history-completeness
    /// declaration.
    ///
    /// # Errors
    ///
    /// Returns an error before mutation when `commit_epoch` is
    /// [`EpochId::PENDING`], the history extends beyond that cut, or the caller
    /// lacks this store's mutation authority.
    #[doc(hidden)]
    pub fn replace_dataset_history_exact(
        &self,
        dataset: RdfDatasetHistory,
        commit_epoch: EpochId,
    ) -> Result<(), String> {
        let mut prepared = self.prepare_dataset_history_replacement(dataset, commit_epoch)?;
        let commit = self.lock_commit_scoped();
        prepared.ready(&commit)?.install().release();
        drop(commit);
        drop(prepared);
        Ok(())
    }

    /// Installs authoritative history while `commit_guard` proves that this
    /// exact store's commit gate is already held by the caller.
    #[doc(hidden)]
    pub fn replace_dataset_history_exact_under_commit_gate(
        &self,
        commit_guard: &RdfCommitGuard<'_>,
        dataset: RdfDatasetHistory,
        commit_epoch: EpochId,
    ) -> Result<(), String> {
        let mut prepared = self.prepare_dataset_history_replacement(dataset, commit_epoch)?;
        prepared.ready(commit_guard)?.install().release();
        Ok(())
    }

    /// Builds an exact detached dataset replacement without changing this store.
    ///
    /// The returned workspace retains the displaced state after installation.
    /// Obtain its ready token under this store's commit gate before publishing
    /// any companion state, and release every publication guard before dropping
    /// the workspace.
    ///
    /// # Errors
    ///
    /// Rejects invalid history/cut combinations and missing mutation authority.
    pub fn prepare_dataset_history_replacement(
        &self,
        dataset: RdfDatasetHistory,
        commit_epoch: EpochId,
    ) -> Result<PreparedRdfDatasetReplacement<'_>, String> {
        if commit_epoch == EpochId::PENDING {
            return Err("RDF dataset history commit epoch cannot be PENDING".to_string());
        }
        for tx in dataset
            .graph_lives()
            .iter()
            .map(RdfGraphLife::tx)
            .chain(dataset.quad_versions().iter().map(RdfQuadVersion::tx))
        {
            if tx.from() > commit_epoch || (!tx.is_open() && tx.to() > commit_epoch) {
                return Err(
                    "RDF dataset history contains a transaction interval beyond its commit epoch"
                        .to_string(),
                );
            }
        }
        if dataset
            .completeness()
            .authoritative_from()
            .is_some_and(|epoch| epoch > commit_epoch)
        {
            return Err(
                "RDF dataset history completeness boundary exceeds its commit epoch".to_string(),
            );
        }
        if !self.unframed_writes_allowed() {
            return Err("RDF dataset history replacement lacks mutation authority".to_string());
        }

        // Restored partitions must share the retained root's allocator and
        // reservation authority. Only installation changes either preimage.
        let allocator = Arc::clone(&self.next_graph_incarnation);
        let reserved = Arc::clone(&self.reserved_graph_incarnations);
        let replacement = Self::with_dataset_identity(
            self.config.clone(),
            dataset.store_id(),
            GraphIncarnationId::DEFAULT_GRAPH,
            Arc::clone(&allocator),
            Arc::clone(&reserved),
        );
        replacement
            .try_set_commit_epoch(commit_epoch)
            .map_err(|error| error.to_string())?;
        *replacement.history_completeness.write() = dataset.completeness();

        let mut named_versions: BTreeMap<(String, GraphIncarnationId), Vec<&RdfQuadVersion>> =
            BTreeMap::new();
        for version in dataset.quad_versions() {
            match version.quad().graph() {
                None => replacement.restore_quad_version(
                    version.quad().triple().clone(),
                    QuadLife {
                        tx: version.tx(),
                        valid: version.valid(),
                    },
                ),
                Some(name) => named_versions
                    .entry((name.to_string(), version.graph_incarnation()))
                    .or_default()
                    .push(version),
            }
        }

        for life in dataset.graph_lives() {
            let name = life
                .graph()
                .name()
                .ok_or_else(|| "RDF named-graph lifecycle has no graph name".to_string())?;
            let incarnation = life.graph().incarnation();
            let graph = Arc::new(Self::with_dataset_identity(
                self.config.clone(),
                dataset.store_id(),
                incarnation,
                Arc::clone(&allocator),
                Arc::clone(&reserved),
            ));
            graph
                .try_set_commit_epoch(commit_epoch)
                .map_err(|error| error.to_string())?;
            for version in named_versions
                .remove(&(name.to_string(), incarnation))
                .unwrap_or_default()
            {
                graph.restore_quad_version(
                    version.quad().triple().clone(),
                    QuadLife {
                        tx: version.tx(),
                        valid: version.valid(),
                    },
                );
            }
            replacement
                .named_graph_history
                .write()
                .push(NamedGraphHistory {
                    name: name.to_string(),
                    incarnation,
                    tx: life.tx(),
                    store: Arc::clone(&graph),
                });
            if life.tx().is_open() {
                replacement
                    .named_graphs
                    .write()
                    .insert(name.to_string(), graph);
            }
        }
        if !named_versions.is_empty() {
            return Err("RDF history contains a quad without a graph lifecycle".to_string());
        }

        let reserved = dataset
            .graph_lives()
            .iter()
            .map(|life| life.graph().incarnation())
            .collect();
        Ok(PreparedRdfDatasetReplacement::new(
            self,
            replacement,
            reserved,
            dataset.next_graph_incarnation().as_u64(),
            commit_epoch,
        ))
    }

    /// Inserts a triple into the store.
    ///
    /// Returns `true` if the triple was newly inserted, `false` if it already existed.
    pub fn insert(&self, triple: Triple) -> bool {
        self.insert_at_current_epoch_with_valid(triple, None)
    }

    fn insert_at_current_epoch_with_valid(
        &self,
        triple: Triple,
        valid: Option<ValidTimeInterval>,
    ) -> bool {
        if !self.unframed_writes_allowed() {
            return false;
        }
        let epoch = self.current_commit_epoch();
        if epoch == EpochId::PENDING {
            return false;
        }
        self.insert_at_epoch_with_valid_inner(triple, epoch, valid)
    }

    /// Tries to insert a triple with explicit real transaction-time and optional
    /// valid-time.
    ///
    /// Used by transaction commit and WAL recovery so history does not depend
    /// on a mutable process-global epoch/validity slot.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for [`EpochId::PENDING`] or an invalid
    /// transaction-state error when this thread lacks the authority for a
    /// sealed store. The live indexes and history are unchanged on error.
    #[doc(hidden)]
    pub fn try_insert_at_epoch_with_valid(
        &self,
        triple: Triple,
        epoch: EpochId,
        valid: Option<ValidTimeInterval>,
    ) -> grafeo_common::utils::error::Result<bool> {
        use grafeo_common::utils::error::{Error, TransactionError};

        if epoch == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "RDF insert epoch cannot be PENDING".to_string(),
            ));
        }
        if !self.unframed_writes_allowed() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF explicit-epoch insert lacks mutation authority".to_string(),
            )));
        }
        Ok(self.insert_at_epoch_with_valid_inner(triple, epoch, valid))
    }

    fn insert_at_epoch_with_valid_inner(
        &self,
        triple: Triple,
        epoch: EpochId,
        valid: Option<ValidTimeInterval>,
    ) -> bool {
        let triple = Arc::new(triple);

        // One primary entry owns both identity and exact source representation.
        {
            let mut triples = self.triples.write();
            match triples.entry(CanonicalTripleKey(Arc::clone(&triple))) {
                hashbrown::hash_map::Entry::Occupied(_) => return false,
                hashbrown::hash_map::Entry::Vacant(entry) => {
                    entry.insert(Arc::clone(&triple));
                }
            }
        }

        // Update indexes
        {
            let mut subject_index = self.subject_index.write();
            subject_index
                .entry(triple.subject().clone())
                .or_default()
                .push(Arc::clone(&triple));
        }

        {
            let mut predicate_index = self.predicate_index.write();
            predicate_index
                .entry(triple.predicate().clone())
                .or_default()
                .push(Arc::clone(&triple));
        }

        if self.config.index_objects {
            let mut object_index = self.object_index.write();
            if let Some(ref mut index) = *object_index {
                index
                    .entry(triple.object().clone())
                    .or_default()
                    .push(Arc::clone(&triple));
            }
        }

        // Update composite indexes
        {
            let mut sp = self.sp_index.write();
            sp.entry((triple.subject().clone(), triple.predicate().clone()))
                .or_default()
                .push(Arc::clone(&triple));
        }

        {
            let mut po = self.po_index.write();
            po.entry((triple.predicate().clone(), triple.object().clone()))
                .or_default()
                .push(Arc::clone(&triple));
        }

        {
            let mut os = self.os_index.write();
            os.entry((triple.object().clone(), triple.subject().clone()))
                .or_default()
                .push(Arc::clone(&triple));
        }

        self.stamp_insert(&triple, epoch, valid);
        self.invalidate_statistics_cache();
        self.lifecycle_revision.fetch_add(1, Ordering::Release);
        true
    }

    /// Inserts a batch of triples with single lock acquisition per index.
    ///
    /// Much more efficient than calling [`Self::insert`] in a loop because each
    /// index lock is acquired once for the entire batch rather than once per
    /// triple (4 lock acquisitions total vs 4 * N).
    ///
    /// Returns the number of triples that were newly inserted (duplicates are
    /// skipped).
    pub fn batch_insert(&self, triples: impl IntoIterator<Item = Triple>) -> usize {
        if !self.unframed_writes_allowed() {
            return 0;
        }
        // Phase 1: deduplicate against primary storage (single lock)
        let mut new_triples = Vec::new();
        {
            let mut primary = self.triples.write();
            for triple in triples {
                let arc = Arc::new(triple);
                let canonical = CanonicalTripleKey(Arc::clone(&arc));
                if let hashbrown::hash_map::Entry::Vacant(entry) = primary.entry(canonical) {
                    entry.insert(Arc::clone(&arc));
                    new_triples.push(arc);
                }
            }
        }

        if new_triples.is_empty() {
            return 0;
        }

        let count = new_triples.len();

        // Phase 2: update subject index (single lock)
        {
            let mut subject_index = self.subject_index.write();
            for triple in &new_triples {
                subject_index
                    .entry(triple.subject().clone())
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 3: update predicate index (single lock)
        {
            let mut predicate_index = self.predicate_index.write();
            for triple in &new_triples {
                predicate_index
                    .entry(triple.predicate().clone())
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 4: update object index if enabled (single lock)
        if self.config.index_objects {
            let mut object_index = self.object_index.write();
            if let Some(ref mut index) = *object_index {
                for triple in &new_triples {
                    index
                        .entry(triple.object().clone())
                        .or_default()
                        .push(Arc::clone(triple));
                }
            }
        }

        // Phase 5: update SP composite index (single lock)
        {
            let mut sp = self.sp_index.write();
            for triple in &new_triples {
                sp.entry((triple.subject().clone(), triple.predicate().clone()))
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 6: update PO composite index (single lock)
        {
            let mut po = self.po_index.write();
            for triple in &new_triples {
                po.entry((triple.predicate().clone(), triple.object().clone()))
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 7: update OS composite index (single lock)
        {
            let mut os = self.os_index.write();
            for triple in &new_triples {
                os.entry((triple.object().clone(), triple.subject().clone()))
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        let epoch = self.current_commit_epoch();
        for triple in &new_triples {
            self.stamp_insert(triple, epoch, None);
        }

        if count > 0 {
            self.invalidate_statistics_cache();
            self.lifecycle_revision.fetch_add(1, Ordering::Release);
        }
        count
    }

    /// Removes a triple from the store.
    ///
    /// Returns `true` if the triple was found and removed.
    pub fn remove(&self, triple: &Triple) -> bool {
        if !self.unframed_writes_allowed() {
            return false;
        }
        let epoch = self.current_commit_epoch();
        if epoch == EpochId::PENDING {
            return false;
        }
        self.remove_at_epoch_inner(triple, epoch)
    }

    /// Tries to remove a triple at an explicit real transaction-time epoch.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for [`EpochId::PENDING`] or an invalid
    /// transaction-state error when this thread lacks the authority for a
    /// sealed store. The live indexes and history are unchanged on error.
    #[doc(hidden)]
    pub fn try_remove_at_epoch(
        &self,
        triple: &Triple,
        epoch: EpochId,
    ) -> grafeo_common::utils::error::Result<bool> {
        use grafeo_common::utils::error::{Error, TransactionError};

        if epoch == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "RDF remove epoch cannot be PENDING".to_string(),
            ));
        }
        if !self.unframed_writes_allowed() {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "RDF explicit-epoch remove lacks mutation authority".to_string(),
            )));
        }
        Ok(self.remove_at_epoch_inner(triple, epoch))
    }

    fn remove_at_epoch_inner(&self, triple: &Triple, epoch: EpochId) -> bool {
        let Some(representative) = self
            .triples
            .write()
            .remove(&CanonicalTripleRef::from(triple))
        else {
            return false;
        };
        // Every secondary index and history record retains this exact spelling.
        let triple = representative.as_ref();

        self.stamp_delete(triple, epoch);

        // Update indexes
        {
            let mut subject_index = self.subject_index.write();
            if let Some(vec) = subject_index.get_mut(triple.subject()) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    subject_index.remove(triple.subject());
                }
            }
        }

        {
            let mut predicate_index = self.predicate_index.write();
            if let Some(vec) = predicate_index.get_mut(triple.predicate()) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    predicate_index.remove(triple.predicate());
                }
            }
        }

        if self.config.index_objects {
            let mut object_index = self.object_index.write();
            if let Some(ref mut index) = *object_index
                && let Some(vec) = index.get_mut(triple.object())
            {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    index.remove(triple.object());
                }
            }
        }

        // Remove from composite indexes
        {
            let mut sp = self.sp_index.write();
            let key = (triple.subject().clone(), triple.predicate().clone());
            if let Some(vec) = sp.get_mut(&key) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    sp.remove(&key);
                }
            }
        }

        {
            let mut po = self.po_index.write();
            let key = (triple.predicate().clone(), triple.object().clone());
            if let Some(vec) = po.get_mut(&key) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    po.remove(&key);
                }
            }
        }

        {
            let mut os = self.os_index.write();
            let key = (triple.object().clone(), triple.subject().clone());
            if let Some(vec) = os.get_mut(&key) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    os.remove(&key);
                }
            }
        }

        self.invalidate_statistics_cache();
        self.lifecycle_revision.fetch_add(1, Ordering::Release);
        true
    }

    /// Returns the number of triples in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.read().len()
    }

    /// Returns `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.read().is_empty()
    }

    /// Estimates the heap footprint of the store for memory reporting.
    ///
    /// Returns `(total_triples, triples_and_indexes_bytes, term_dictionary_bytes,
    /// ring_index_bytes, named_graph_count)`. Recurses into named graphs so the
    /// totals reflect the full RDF memory, not just the default graph.
    ///
    /// The index-bytes figure is an approximation: each `HashMap` entry is
    /// charged for a pointer-sized key plus capacity-based Vec overhead. It
    /// undercounts the per-`Term` heap (a `Term::IRI(String)` carries its
    /// payload) and overcounts hash-map empty buckets. Good enough to surface
    /// "RDF is eating the heap" in an introspection breakdown.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize, usize, usize) {
        use std::mem::size_of;

        let triples = self.triples.read();
        let triple_arc_bytes = triples.capacity()
            * (size_of::<CanonicalTripleKey>() + size_of::<Arc<Triple>>() + size_of::<u64>());
        let triple_payload_bytes = triples.len() * size_of::<Triple>();
        drop(triples);

        let index_entry = size_of::<(Term, Vec<Arc<Triple>>)>();
        let composite_entry = size_of::<((Term, Term), Vec<Arc<Triple>>)>();

        let subject_bytes = {
            let g = self.subject_index.read();
            g.capacity() * index_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let predicate_bytes = {
            let g = self.predicate_index.read();
            g.capacity() * index_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let object_bytes = self.object_index.read().as_ref().map_or(0, |g| {
            g.capacity() * index_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        });
        let sp_bytes = {
            let g = self.sp_index.read();
            g.capacity() * composite_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let po_bytes = {
            let g = self.po_index.read();
            g.capacity() * composite_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let os_bytes = {
            let g = self.os_index.read();
            g.capacity() * composite_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };

        let mut triples_and_indexes = triple_arc_bytes
            + triple_payload_bytes
            + subject_bytes
            + predicate_bytes
            + object_bytes
            + sp_bytes
            + po_bytes
            + os_bytes;

        let term_dict_bytes = self
            .dictionary_cache
            .read()
            .as_ref()
            .map_or(0, |d| d.len() * (size_of::<Term>() + size_of::<u32>()) * 2);

        #[cfg(feature = "ring-index")]
        let ring_bytes = self
            .ring
            .read()
            .as_ref()
            .map_or(0, |r| r.len() * size_of::<(u32, u32, u32)>());
        #[cfg(not(feature = "ring-index"))]
        let ring_bytes = 0usize;

        let named_graphs_guard = self.named_graphs.read();
        let named_graph_count = named_graphs_guard.len();
        let mut total_triples = self.len();
        let mut total_term_dict = term_dict_bytes;
        let mut total_ring = ring_bytes;
        for (_name, graph) in named_graphs_guard.iter() {
            let (ng_triples, ng_store, ng_dict, ng_ring, _) = graph.heap_memory_bytes();
            total_triples += ng_triples;
            triples_and_indexes += ng_store;
            total_term_dict += ng_dict;
            total_ring += ng_ring;
        }

        (
            total_triples,
            triples_and_indexes,
            total_term_dict,
            total_ring,
            named_graph_count,
        )
    }

    /// Checks if a triple exists in the store.
    #[must_use]
    pub fn contains(&self, triple: &Triple) -> bool {
        self.triples
            .read()
            .contains_key(&CanonicalTripleRef::from(triple))
    }

    /// Returns all triples in the store.
    pub fn triples(&self) -> Vec<Arc<Triple>> {
        self.triples.read().values().cloned().collect()
    }

    /// Returns triples matching the given pattern.
    ///
    /// Uses composite indexes for 2-bound and 3-bound queries (O(1) lookup),
    /// single-term indexes for 1-bound queries, and full scan for unbound.
    pub fn find(&self, pattern: &TriplePattern) -> Vec<Arc<Triple>> {
        if let (Some(subject), Some(predicate), Some(object)) =
            (&pattern.subject, &pattern.predicate, &pattern.object)
        {
            let key = CanonicalTripleRef([subject, predicate, object]);
            return self.triples.read().get(&key).cloned().into_iter().collect();
        }
        // Exact-term indexes preserve source language-tag spelling. Narrow by
        // exact S/P indexes when possible, then compare the object using RDF's
        // case-insensitive language identity.
        if pattern.object.as_ref().is_some_and(|object| {
            object
                .as_literal()
                .is_some_and(|literal| literal.language().is_some())
        }) {
            let candidates = match (&pattern.subject, &pattern.predicate) {
                (Some(subject), Some(predicate)) => self
                    .sp_index
                    .read()
                    .get(&(subject.clone(), predicate.clone()))
                    .cloned()
                    .unwrap_or_default(),
                (Some(subject), None) => self
                    .subject_index
                    .read()
                    .get(subject)
                    .cloned()
                    .unwrap_or_default(),
                (None, Some(predicate)) => self
                    .predicate_index
                    .read()
                    .get(predicate)
                    .cloned()
                    .unwrap_or_default(),
                (None, None) => self.triples.read().values().cloned().collect(),
            };
            return candidates
                .into_iter()
                .filter(|triple| pattern.matches(triple))
                .collect();
        }
        match (&pattern.subject, &pattern.predicate, &pattern.object) {
            // 3-bound: use SP composite, filter on O (at most 1 result)
            (Some(s), Some(p), Some(o)) => {
                let index = self.sp_index.read();
                if let Some(triples) = index.get(&(s.clone(), p.clone())) {
                    triples
                        .iter()
                        .filter(|t| t.object() == o)
                        .cloned()
                        .collect()
                } else {
                    Vec::new()
                }
            }
            // 2-bound: use composite indexes (O(1) lookup, no filtering)
            (Some(s), Some(p), None) => {
                let index = self.sp_index.read();
                index
                    .get(&(s.clone(), p.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            (Some(s), None, Some(o)) => {
                let index = self.os_index.read();
                index
                    .get(&(o.clone(), s.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            (None, Some(p), Some(o)) => {
                let index = self.po_index.read();
                index
                    .get(&(p.clone(), o.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            // 1-bound: use single-term indexes
            (Some(s), None, None) => {
                let index = self.subject_index.read();
                index.get(s).cloned().unwrap_or_default()
            }
            (None, Some(p), None) => {
                let index = self.predicate_index.read();
                index.get(p).cloned().unwrap_or_default()
            }
            (None, None, Some(o)) if self.config.index_objects => {
                let index = self.object_index.read();
                if let Some(ref idx) = *index {
                    idx.get(o).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                }
            }
            // 0-bound or O-only without object index: full scan
            _ => self
                .triples
                .read()
                .values()
                .filter(|t| pattern.matches(t))
                .cloned()
                .collect(),
        }
    }

    /// Returns triples with the given subject.
    pub fn triples_with_subject(&self, subject: &Term) -> Vec<Arc<Triple>> {
        let index = self.subject_index.read();
        index.get(subject).cloned().unwrap_or_default()
    }

    /// Returns triples with the given predicate.
    pub fn triples_with_predicate(&self, predicate: &Term) -> Vec<Arc<Triple>> {
        let index = self.predicate_index.read();
        index.get(predicate).cloned().unwrap_or_default()
    }

    /// Returns triples with the given object.
    pub fn triples_with_object(&self, object: &Term) -> Vec<Arc<Triple>> {
        if object
            .as_literal()
            .is_some_and(|literal| literal.language().is_some())
        {
            return self.find(&TriplePattern {
                subject: None,
                predicate: None,
                object: Some(object.clone()),
            });
        }
        let index = self.object_index.read();
        if let Some(ref idx) = *index {
            idx.get(object).cloned().unwrap_or_default()
        } else {
            // Fall back to full scan if object index is disabled
            self.triples
                .read()
                .values()
                .filter(|t| t.object() == object)
                .cloned()
                .collect()
        }
    }

    /// Returns all unique subjects in the store.
    pub fn subjects(&self) -> Vec<Term> {
        self.subject_index.read().keys().cloned().collect()
    }

    /// Returns all unique predicates in the store.
    pub fn predicates(&self) -> Vec<Term> {
        self.predicate_index.read().keys().cloned().collect()
    }

    /// Returns all unique objects in the store.
    pub fn objects(&self) -> Vec<Term> {
        if self.config.index_objects {
            let index = self.object_index.read();
            if let Some(ref idx) = *index {
                return idx.keys().cloned().collect();
            }
        }
        // Fall back to collecting from triples
        let triples = self.triples.read();
        let mut objects = FxHashSet::default();
        for triple in triples.values() {
            objects.insert(triple.object().clone());
        }
        objects.into_iter().collect()
    }

    /// Clears all triples from the store.
    pub fn clear(&self) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let removed: Vec<Arc<Triple>> = self.triples.read().values().cloned().collect();
        let changed = !removed.is_empty();
        self.triples.write().clear();
        self.subject_index.write().clear();
        self.predicate_index.write().clear();
        if let Some(ref mut idx) = *self.object_index.write() {
            idx.clear();
        }
        self.sp_index.write().clear();
        self.po_index.write().clear();
        self.os_index.write().clear();
        let epoch = self.current_commit_epoch();
        for triple in removed {
            self.stamp_delete(&triple, epoch);
        }
        self.invalidate_statistics_cache();
        if changed {
            self.lifecycle_revision.fetch_add(1, Ordering::Release);
        }
    }

    /// Returns store statistics.
    #[must_use]
    pub fn stats(&self) -> RdfStoreStats {
        RdfStoreStats {
            triple_count: self.len(),
            subject_count: self.subject_index.read().len(),
            predicate_count: self.predicate_index.read().len(),
            object_count: if self.config.index_objects {
                self.object_index.read().as_ref().map_or(0, |i| i.len())
            } else {
                0
            },
            graph_count: self.named_graphs.read().len(),
        }
    }

    /// Collects detailed RDF statistics for query optimization.
    ///
    /// Iterates all triples to compute per-predicate cardinality estimates,
    /// distinct subject/object counts, and index access pattern costs.
    #[must_use]
    pub fn collect_statistics(&self) -> crate::statistics::RdfStatistics {
        let mut collector = crate::statistics::RdfStatisticsCollector::new();
        let triples = self.triples.read();
        for triple in triples.values() {
            collector.record_triple(
                &triple.subject().to_string(),
                &triple.predicate().to_string(),
                &triple.object().to_string(),
            );
        }
        collector.build()
    }

    /// Returns cached RDF statistics, computing them on first call.
    ///
    /// The cache is invalidated by any mutation (insert, delete, bulk load).
    /// This avoids the full-table-scan overhead of `collect_statistics()` on
    /// every query, which was a measurable regression for larger datasets.
    #[must_use]
    pub fn get_or_collect_statistics(&self) -> Arc<crate::statistics::RdfStatistics> {
        // Fast path: return cached statistics if available.
        if let Some(cached) = self.statistics_cache.read().as_ref() {
            return Arc::clone(cached);
        }

        // Slow path: compute and cache.
        let stats = Arc::new(self.collect_statistics());
        *self.statistics_cache.write() = Some(Arc::clone(&stats));
        stats
    }

    /// Invalidates the cached RDF statistics and term dictionary. Called after any mutation.
    fn invalidate_statistics_cache(&self) {
        *self.statistics_cache.write() = None;
        *self.dictionary_cache.write() = None;
        #[cfg(feature = "ring-index")]
        {
            self.ring_stale
                .store(true, std::sync::atomic::Ordering::Relaxed);
            *self.ring.write() = None;
        }
    }

    /// Returns a cached term dictionary, building it on first call.
    ///
    /// The dictionary maps each unique `Term` in the store to a compact `u32` ID.
    /// It is invalidated by any mutation (insert, delete, bulk load).
    #[must_use]
    pub fn get_or_build_dictionary(&self) -> Arc<super::dictionary::TermDictionary> {
        // Fast path: return cached dictionary
        {
            let cache = self.dictionary_cache.read();
            if let Some(dict) = cache.as_ref() {
                return Arc::clone(dict);
            }
        }

        // Slow path: build dictionary from all triples
        let triples = self.triples.read();
        let mut dict =
            super::dictionary::TermDictionary::with_capacity(triples.len().saturating_mul(3));
        for triple in triples.values() {
            dict.get_or_insert(triple.subject());
            dict.get_or_insert(triple.predicate());
            dict.get_or_insert(triple.object());
        }

        let dict = Arc::new(dict);
        *self.dictionary_cache.write() = Some(Arc::clone(&dict));
        dict
    }

    /// Returns the cached term dictionary if it exists, without building it.
    #[must_use]
    pub fn term_dictionary(&self) -> Option<Arc<super::dictionary::TermDictionary>> {
        self.dictionary_cache.read().clone()
    }

    /// Returns the Ring Index if it exists and is not stale.
    #[cfg(feature = "ring-index")]
    #[must_use]
    pub fn ring(&self) -> Option<Arc<crate::index::ring::TripleRing>> {
        if self.ring_stale.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        self.ring.read().clone()
    }

    /// Builds or rebuilds the Ring Index from current triples.
    ///
    /// The Ring provides ~3x memory reduction and O(log sigma) pattern counting.
    /// It is automatically built during `bulk_load()` and can be rebuilt
    /// explicitly after incremental mutations. This is a safe derived-cache
    /// rebuild from the already-authorized current triple set; it cannot inject
    /// caller-supplied logical state and therefore remains available on a
    /// sealed read handle.
    #[cfg(feature = "ring-index")]
    pub fn rebuild_ring(&self) {
        let triples = self.triples.read();
        if triples.is_empty() {
            *self.ring.write() = None;
            return;
        }
        let ring = crate::index::ring::TripleRing::from_triples(
            triples.values().map(|t| t.as_ref().clone()),
        );
        *self.ring.write() = Some(Arc::new(ring));
        self.ring_stale
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Sets the Ring Index directly (used during container deserialization).
    #[cfg(feature = "ring-index")]
    pub fn set_ring(&self, ring: crate::index::ring::TripleRing) {
        if !self.unframed_writes_allowed() {
            return;
        }
        *self.ring.write() = Some(Arc::new(ring));
        self.ring_stale
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    // =========================================================================
    // Bulk loading
    // =========================================================================

    /// Loads triples in bulk, replacing all existing data.
    ///
    /// Much faster than `batch_insert()` for initial data loading:
    /// - Deduplicates canonical aliases, retaining the first source spelling
    /// - Builds all indexes in a single pass using pre-sized `HashMap`s
    /// - Computes [`RdfStatistics`](crate::statistics::RdfStatistics) during the
    ///   same traversal (no extra scan needed)
    ///
    /// **Warning**: This replaces all existing triples and indexes in the store.
    /// Any previously stored data will be lost.
    pub fn bulk_load(&self, triples: impl IntoIterator<Item = Triple>) -> BulkLoadResult {
        if !self.unframed_writes_allowed() {
            return BulkLoadResult {
                triple_count: 0,
                statistics: crate::statistics::RdfStatistics::new(),
            };
        }
        let input = triples.into_iter();
        let mut primary = HashMap::with_capacity(input.size_hint().0);
        let arcs: Vec<Arc<Triple>> = input
            .filter_map(|triple| {
                let triple = Arc::new(triple);
                match primary.entry(CanonicalTripleKey(Arc::clone(&triple))) {
                    hashbrown::hash_map::Entry::Occupied(_) => None,
                    hashbrown::hash_map::Entry::Vacant(entry) => {
                        entry.insert(Arc::clone(&triple));
                        Some(triple)
                    }
                }
            })
            .collect();
        let count = arcs.len();

        if count == 0 {
            self.clear();
            self.history.write().clear();
            return BulkLoadResult {
                triple_count: 0,
                statistics: crate::statistics::RdfStatistics::new(),
            };
        }

        // Build all indexes in local HashMaps (no locks during build)
        let hasher = || foldhash::fast::RandomState::default();
        let mut subject_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 4, hasher());
        let mut predicate_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 8, hasher());
        let mut object_idx_map = hashbrown::HashMap::with_capacity_and_hasher(count / 4, hasher());
        let mut sp_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 2, hasher());
        let mut po_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 2, hasher());
        let mut os_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 2, hasher());

        let mut stats_collector = crate::statistics::RdfStatisticsCollector::new();

        for triple in &arcs {
            // Single-term indexes
            subject_idx
                .entry(triple.subject().clone())
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            predicate_idx
                .entry(triple.predicate().clone())
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            if self.config.index_objects {
                object_idx_map
                    .entry(triple.object().clone())
                    .or_insert_with(Vec::new)
                    .push(Arc::clone(triple));
            }

            // Composite indexes
            sp_idx
                .entry((triple.subject().clone(), triple.predicate().clone()))
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            po_idx
                .entry((triple.predicate().clone(), triple.object().clone()))
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            os_idx
                .entry((triple.object().clone(), triple.subject().clone()))
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));

            // Collect statistics in the same pass
            stats_collector.record_triple(
                &triple.subject().to_string(),
                &triple.predicate().to_string(),
                &triple.object().to_string(),
            );
        }

        // Swap indexes into the store (one lock acquisition per index)
        let epoch = self.current_commit_epoch();
        let history = primary
            .values()
            .map(|triple| {
                (
                    Arc::clone(triple),
                    vec![QuadLife {
                        tx: EpochInterval::open(epoch),
                        valid: None,
                    }],
                )
            })
            .collect();
        *self.triples.write() = primary;
        *self.subject_index.write() = subject_idx;
        *self.predicate_index.write() = predicate_idx;
        *self.object_index.write() = if self.config.index_objects {
            Some(object_idx_map)
        } else {
            None
        };
        *self.sp_index.write() = sp_idx;
        *self.po_index.write() = po_idx;
        *self.os_index.write() = os_idx;
        *self.history.write() = history;

        let stats = stats_collector.build();
        // Cache the freshly-computed statistics from the bulk load pass.
        *self.statistics_cache.write() = Some(Arc::new(stats.clone()));
        // Invalidate the dictionary cache: the old dictionary mapped terms
        // from the previous dataset, not the newly loaded one.
        *self.dictionary_cache.write() = None;

        // Build Ring Index automatically during bulk load (when feature is enabled).
        #[cfg(feature = "ring-index")]
        {
            let ring = crate::index::ring::TripleRing::from_triples(
                self.triples.read().values().map(|t| t.as_ref().clone()),
            );
            *self.ring.write() = Some(Arc::new(ring));
            self.ring_stale
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }

        self.lifecycle_revision.fetch_add(1, Ordering::Release);

        BulkLoadResult {
            triple_count: count,
            statistics: stats,
        }
    }

    /// Parses and loads an N-Triples document, replacing all existing data.
    ///
    /// Each line is parsed as `<subject> <predicate> <object> .` per the
    /// [N-Triples spec](https://www.w3.org/TR/n-triples/). Empty lines and
    /// comment lines (starting with `#`) are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or if a line cannot be parsed.
    pub fn load_ntriples(
        &self,
        reader: impl std::io::BufRead,
    ) -> Result<BulkLoadResult, NTriplesError> {
        if !self.unframed_writes_allowed() {
            return Ok(BulkLoadResult {
                triple_count: 0,
                statistics: crate::statistics::RdfStatistics::new(),
            });
        }
        let mut triples = Vec::new();
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(NTriplesError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let triple = parse_ntriples_line(trimmed).ok_or_else(|| NTriplesError::Parse {
                line: line_no + 1,
                content: line.clone(),
            })?;
            triples.push(triple);
        }
        Ok(self.bulk_load(triples))
    }

    /// Parses and loads a Turtle document, replacing all existing data.
    ///
    /// # Errors
    ///
    /// Returns a `TurtleError` on parse failure.
    pub fn load_turtle(&self, input: &str) -> Result<BulkLoadResult, super::turtle::TurtleError> {
        if !self.unframed_writes_allowed() {
            return Ok(BulkLoadResult {
                triple_count: 0,
                statistics: crate::statistics::RdfStatistics::new(),
            });
        }
        let triples = super::turtle::TurtleParser::new().parse(input)?;
        Ok(self.bulk_load(triples))
    }

    /// Parses a Turtle document and streams triples into the store via batched inserts.
    ///
    /// Unlike [`load_turtle`](Self::load_turtle), this does not replace existing data.
    /// Triples are inserted incrementally in batches, keeping memory bounded regardless
    /// of document size. Duplicate triples are skipped during each batch insert.
    ///
    /// # Errors
    ///
    /// Returns a `TurtleError` on parse failure.
    pub fn load_turtle_streaming(
        &self,
        input: &str,
        batch_size: usize,
    ) -> Result<usize, super::turtle::TurtleError> {
        if !self.unframed_writes_allowed() {
            return Ok(0);
        }
        let mut sink = super::sink::BatchInsertSink::new(self, batch_size);
        let mut parser = super::turtle::TurtleParser::new();
        parser.parse_into(input, &mut sink)?;
        Ok(sink.total_inserted())
    }

    /// Reads a Turtle document from a buffered reader and streams triples into the store.
    ///
    /// The entire reader is consumed into a string first (Turtle requires random access
    /// for prefix resolution), then parsed with batched inserts. For pure streaming from
    /// a reader without replacing existing data, this is the recommended entry point.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading fails, or a `TurtleError` on parse failure.
    pub fn load_turtle_reader(
        &self,
        reader: impl std::io::Read,
        batch_size: usize,
    ) -> Result<usize, NTriplesError> {
        if !self.unframed_writes_allowed() {
            return Ok(0);
        }
        let mut input = String::new();
        std::io::Read::read_to_string(&mut { reader }, &mut input).map_err(NTriplesError::Io)?;
        self.load_turtle_streaming(&input, batch_size)
            .map_err(|e| NTriplesError::Parse {
                line: e.line,
                content: e.message,
            })
    }

    /// Parses N-Triples from a reader, streaming triples into the store via batched inserts.
    ///
    /// Unlike [`load_ntriples`](Self::load_ntriples), this does not replace existing data.
    /// Triples are inserted incrementally in batches, keeping memory bounded.
    ///
    /// # Errors
    ///
    /// Returns an `NTriplesError` on I/O or parse failure.
    pub fn load_ntriples_streaming(
        &self,
        reader: impl std::io::BufRead,
        batch_size: usize,
    ) -> Result<usize, NTriplesError> {
        if !self.unframed_writes_allowed() {
            return Ok(0);
        }
        let mut sink = super::sink::BatchInsertSink::new(self, batch_size);
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(NTriplesError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let triple = parse_ntriples_line(trimmed).ok_or_else(|| NTriplesError::Parse {
                line: line_no + 1,
                content: line.clone(),
            })?;
            sink.emit(triple).map_err(|e| NTriplesError::Parse {
                line: line_no + 1,
                content: e,
            })?;
        }
        sink.finish().map_err(|e| NTriplesError::Parse {
            line: 0,
            content: e,
        })?;
        Ok(sink.total_inserted())
    }

    /// Serializes this store's triples to Turtle format.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if serialization fails.
    pub fn to_turtle(&self) -> std::io::Result<String> {
        super::turtle::TurtleSerializer::new().to_string(&self.triples())
    }

    /// Serializes this store (default + named graphs) to N-Quads format.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if serialization fails.
    pub fn to_nquads(&self) -> std::io::Result<String> {
        super::nquads::to_nquads_string(self)
    }

    // =========================================================================
    // Named graph support
    // =========================================================================

    fn inherit_transaction_snapshots(&self, child: &RdfStore) {
        let snapshots: Vec<(TransactionId, EpochId)> = self
            .tx_buffer
            .read()
            .snapshot_epochs
            .iter()
            .map(|(tid, epoch)| (*tid, *epoch))
            .collect();
        if snapshots.is_empty() {
            return;
        }
        let mut child_buffer = child.tx_buffer.write();
        for (tid, epoch) in snapshots {
            child_buffer.snapshot_epochs.entry(tid).or_insert(epoch);
        }
    }

    /// Returns a named graph by IRI, or `None` if it doesn't exist.
    #[must_use]
    pub fn graph(&self, name: &str) -> Option<Arc<RdfStore>> {
        let g = self.named_graphs.read().get(name).cloned()?;
        self.prepare_named_graph_for_access(&g);
        Some(g)
    }

    fn prepare_named_graph_for_access(&self, graph: &Arc<RdfStore>) {
        self.inherit_transaction_snapshots(graph);
        let scope = self
            .mutation_scope
            .load(std::sync::atomic::Ordering::Acquire);
        if scope != 0 {
            let _ = graph.seal_with_scope(scope);
        }
    }

    fn resolve_named_graph_locked(
        &self,
        buffer: &mut TransactionBuffer,
        name: &str,
        transaction_id: TransactionId,
    ) -> Option<Arc<RdfStore>> {
        if let Some(store) = buffer
            .created_graphs
            .get(&transaction_id)
            .and_then(|graphs| graphs.get(name))
        {
            return Some(Arc::clone(store));
        }
        if buffer
            .dropped_graphs
            .get(&transaction_id)
            .is_some_and(|graphs| graphs.contains_key(name))
        {
            return None;
        }

        // Read Committed intentionally resolves the live registry on every
        // statement. Only transactions with a registered start-epoch cut get
        // repeatable named-graph identity/existence.
        if !buffer.snapshot_epochs.contains_key(&transaction_id) {
            return self.named_graphs.read().get(name).cloned();
        }
        if let Some(pinned) = buffer
            .read_graphs
            .get(&transaction_id)
            .and_then(|graphs| graphs.get(name))
        {
            return pinned.clone();
        }

        let committed = if buffer.snapshotted_graph_catalogs.contains(&transaction_id) {
            None
        } else {
            self.named_graphs.read().get(name).cloned()
        };
        buffer
            .read_graphs
            .entry(transaction_id)
            .or_default()
            .insert(name.to_string(), committed.clone());
        committed
    }

    /// Returns the named-graph partition visible to one transaction.
    ///
    /// Detached creates are visible only to their owner; staged drops are
    /// hidden only from their owner until commit publication.
    #[must_use]
    pub fn graph_in_transaction(
        &self,
        name: &str,
        transaction_id: Option<TransactionId>,
    ) -> Option<Arc<RdfStore>> {
        let Some(transaction_id) = transaction_id else {
            return self.graph(name);
        };
        let graph =
            self.resolve_named_graph_locked(&mut self.tx_buffer.write(), name, transaction_id)?;
        self.prepare_named_graph_for_access(&graph);
        Some(graph)
    }

    /// Resolves a named graph for transaction-local mutation without creating it.
    ///
    /// The first write pins the exact shared partition and its revision. Later
    /// writes in the same transaction keep using that partition even under Read
    /// Committed, so lifecycle validation can reject a concurrent DROP/CREATE
    /// instead of publishing writes into a replacement incarnation. Detached
    /// graphs created by this transaction are returned without a shared pin.
    #[doc(hidden)]
    #[must_use]
    pub fn graph_for_mutation_in_tx(
        &self,
        name: &str,
        transaction_id: TransactionId,
    ) -> Option<Arc<RdfStore>> {
        if !self.unframed_writes_allowed() {
            return None;
        }

        let graph = {
            let mut buffer = self.tx_buffer.write();
            if let Some(created) = buffer
                .created_graphs
                .get(&transaction_id)
                .and_then(|graphs| graphs.get(name))
                .cloned()
            {
                created
            } else if buffer
                .dropped_graphs
                .get(&transaction_id)
                .is_some_and(|graphs| graphs.contains_key(name))
            {
                return None;
            } else if let Some(pinned) = buffer
                .touched_graphs
                .get(&transaction_id)
                .and_then(|graphs| graphs.get(name))
                .cloned()
            {
                pinned.store
            } else {
                let graph = self.resolve_named_graph_locked(&mut buffer, name, transaction_id)?;
                buffer
                    .touched_graphs
                    .entry(transaction_id)
                    .or_default()
                    .insert(
                        name.to_string(),
                        RdfGraphPin {
                            revision: graph.lifecycle_revision.load(Ordering::Acquire),
                            store: Arc::clone(&graph),
                        },
                    );
                graph
            }
        };
        self.prepare_named_graph_for_access(&graph);
        Some(graph)
    }

    fn new_detached_graph(&self) -> Result<Arc<RdfStore>, RdfHistoryError> {
        // Reserve the number and capture its dataset identity under the same
        // writer used by exact replacement; never increment before admission.
        let mut reserved = self.reserved_graph_incarnations.write();
        let incarnation = self
            .next_graph_incarnation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| RdfHistoryError::GraphIncarnationExhausted)?;
        let child = RdfStore::with_dataset_identity(
            self.config.clone(),
            self.store_id(),
            GraphIncarnationId::new(incarnation),
            Arc::clone(&self.next_graph_incarnation),
            Arc::clone(&self.reserved_graph_incarnations),
        );
        reserved.insert(GraphIncarnationId::new(incarnation));
        child
            .commit_epoch
            .store(self.commit_epoch.load(Ordering::Acquire), Ordering::Release);
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0 {
            let _ = child.seal_with_scope(scope);
        }
        let child = Arc::new(child);
        // Snapshot inheritance has its own transaction-buffer lock order.
        drop(reserved);
        self.inherit_transaction_snapshots(&child);
        Ok(child)
    }

    fn detached_graph_with_incarnation(
        &self,
        incarnation: GraphIncarnationId,
    ) -> Result<Arc<RdfStore>, String> {
        let next = incarnation
            .checked_next()
            .ok_or_else(|| "RDF named-graph incarnation space exhausted".to_string())?;
        if incarnation.is_default_graph() {
            return Err("named RDF graph cannot use the default incarnation".to_string());
        }
        let child = RdfStore::with_dataset_identity(
            self.config.clone(),
            self.store_id(),
            incarnation,
            Arc::clone(&self.next_graph_incarnation),
            Arc::clone(&self.reserved_graph_incarnations),
        );
        child
            .commit_epoch
            .store(self.commit_epoch.load(Ordering::Acquire), Ordering::Release);
        let scope = self.mutation_scope.load(Ordering::Acquire);
        if scope != 0 && !child.seal_with_scope(scope) {
            return Err("failed to apply RDF graph mutation scope".to_string());
        }
        self.next_graph_incarnation
            .fetch_max(next.as_u64(), Ordering::AcqRel);
        let child = Arc::new(child);
        self.inherit_transaction_snapshots(&child);
        Ok(child)
    }

    /// Recovers an exact named-graph creation without allocating a new
    /// incarnation.
    ///
    /// Duplicate replay of the same live incarnation is idempotent. Reusing an
    /// incarnation for another lifetime or colliding with a different active
    /// incarnation fails closed.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identity, authority, exhaustion, or an
    /// incarnation/name conflict.
    #[doc(hidden)]
    pub fn create_graph_with_incarnation_at(
        &self,
        name: &str,
        incarnation: GraphIncarnationId,
        epoch: EpochId,
    ) -> Result<bool, String> {
        if !self.unframed_writes_allowed() {
            return Err("RDF graph recovery lacks mutation authority".to_string());
        }
        RdfGraphIdentity::named(name.to_string(), incarnation)
            .map_err(|error| error.to_string())?;
        if epoch == EpochId::PENDING {
            return Err("RDF graph recovery cannot use the pending epoch".to_string());
        }
        let _lifecycle = self.history_lifecycle_lock.lock();
        if let Some(active) = self.named_graphs.read().get(name) {
            return if active.graph_incarnation() == incarnation {
                Ok(false)
            } else {
                Err(format!(
                    "RDF graph <{name}> is incarnation {}, not recovered incarnation {incarnation}",
                    active.graph_incarnation()
                ))
            };
        }
        if !self.reserved_graph_incarnations.write().insert(incarnation) {
            return Err(format!(
                "RDF graph incarnation {incarnation} was already used"
            ));
        }
        self.commit_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        let child = match self.detached_graph_with_incarnation(incarnation) {
            Ok(child) => child,
            Err(error) => {
                self.reserved_graph_incarnations
                    .write()
                    .remove(&incarnation);
                return Err(error);
            }
        };
        child.commit_epoch.store(epoch.as_u64(), Ordering::Release);
        self.named_graphs
            .write()
            .insert(name.to_string(), Arc::clone(&child));
        self.record_graph_create(name, &child, epoch);
        Ok(true)
    }

    /// Recovers an exact named-graph drop and verifies the active incarnation.
    ///
    /// # Errors
    ///
    /// Returns an error for missing authority, a pending epoch, or a graph name
    /// currently owned by a different incarnation.
    #[doc(hidden)]
    pub fn drop_graph_with_incarnation_at(
        &self,
        name: &str,
        incarnation: GraphIncarnationId,
        epoch: EpochId,
    ) -> Result<bool, String> {
        if !self.unframed_writes_allowed() {
            return Err("RDF graph recovery lacks mutation authority".to_string());
        }
        if epoch == EpochId::PENDING {
            return Err("RDF graph recovery cannot use the pending epoch".to_string());
        }
        let _lifecycle = self.history_lifecycle_lock.lock();
        let Some(active) = self.named_graphs.read().get(name).cloned() else {
            let already_dropped = self.named_graph_history.read().iter().any(|life| {
                life.name == name && life.incarnation == incarnation && !life.tx.is_open()
            });
            return if already_dropped {
                Ok(false)
            } else {
                Err(format!(
                    "cannot recover DROP of absent RDF graph <{name}> incarnation {incarnation}"
                ))
            };
        };
        if active.graph_incarnation() != incarnation {
            return Err(format!(
                "cannot recover DROP of RDF graph <{name}> incarnation {incarnation}; active incarnation is {}",
                active.graph_incarnation()
            ));
        }
        self.commit_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        self.named_graphs.write().remove(name);
        self.record_graph_drop(name, &active, epoch);
        self.lifecycle_revision.fetch_add(1, Ordering::Release);
        Ok(true)
    }

    /// Returns a named graph only when its exact incarnation is active.
    #[doc(hidden)]
    #[must_use]
    pub fn graph_with_incarnation(
        &self,
        name: &str,
        incarnation: GraphIncarnationId,
    ) -> Option<Arc<RdfStore>> {
        self.named_graphs
            .read()
            .get(name)
            .filter(|graph| graph.graph_incarnation() == incarnation)
            .cloned()
    }

    fn record_graph_create(&self, name: &str, graph: &Arc<RdfStore>, epoch: EpochId) {
        self.named_graph_history.write().push(NamedGraphHistory {
            name: name.to_string(),
            incarnation: graph.graph_incarnation,
            tx: EpochInterval::open(epoch),
            store: Arc::clone(graph),
        });
    }

    fn record_graph_drop(&self, name: &str, graph: &Arc<RdfStore>, epoch: EpochId) {
        graph.close_all_history_at(epoch);
        let mut history = self.named_graph_history.write();
        if let Some(index) = history.iter().rposition(|entry| {
            entry.name == name
                && entry.tx.is_open()
                && entry.incarnation == graph.graph_incarnation
                && Arc::ptr_eq(&entry.store, graph)
        }) {
            if history[index].tx.from() == epoch {
                history.remove(index);
            } else {
                let from = history[index].tx.from();
                history[index].tx = EpochInterval::closed(from, epoch);
            }
        }
    }

    fn close_all_history_at(&self, epoch: EpochId) {
        let mut history = self.history.write();
        history.retain(|_, versions| {
            if let Some(last) = versions.last_mut()
                && last.tx.is_open()
            {
                if last.tx.from() == epoch {
                    versions.pop();
                } else {
                    last.tx = EpochInterval::closed(last.tx.from(), epoch);
                }
            }
            !versions.is_empty()
        });
    }

    /// Returns a named graph, creating it if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`RdfHistoryError::GraphIncarnationExhausted`] when the durable
    /// incarnation namespace has no value left to reserve.
    pub fn graph_or_create(&self, name: &str) -> Result<Arc<RdfStore>, RdfHistoryError> {
        self.graph_or_create_in_tx(name, None)
    }

    /// Like [`graph_or_create`](Self::graph_or_create), recording a new graph on `tid` so
    /// rollback can drop it if it stays empty.
    ///
    /// # Errors
    ///
    /// Returns [`RdfHistoryError::GraphIncarnationExhausted`] when the durable
    /// incarnation namespace has no value left to reserve.
    pub fn graph_or_create_in_tx(
        &self,
        name: &str,
        tid: Option<TransactionId>,
    ) -> Result<Arc<RdfStore>, RdfHistoryError> {
        // Retain lifecycle admission from identity allocation through shared
        // or transaction-local publication. A detached old-identity child must
        // never wait across an exact dataset replacement before insertion.
        let _lifecycle = self.history_lifecycle_lock.lock();
        let Some(transaction_id) = tid else {
            if let Some(graph) = self.graph(name) {
                return Ok(graph);
            }
            let graph = self.new_detached_graph()?;
            if !self.unframed_writes_allowed() {
                return Ok(graph);
            }
            let mut graphs = self.named_graphs.write();
            if let Some(existing) = graphs.get(name) {
                return Ok(Arc::clone(existing));
            }
            graphs.insert(name.to_string(), Arc::clone(&graph));
            self.record_graph_create(name, &graph, self.current_commit_epoch());
            return Ok(graph);
        };

        if let Some(graph) = self
            .tx_buffer
            .read()
            .created_graphs
            .get(&transaction_id)
            .and_then(|graphs| graphs.get(name))
            .cloned()
        {
            return Ok(graph);
        }
        let staged_drop = self
            .tx_buffer
            .read()
            .dropped_graphs
            .get(&transaction_id)
            .is_some_and(|graphs| graphs.contains_key(name));
        if !staged_drop {
            let existing = if self.unframed_writes_allowed() {
                self.graph_for_mutation_in_tx(name, transaction_id)
            } else {
                // Preserve the existing read result for a sealed raw caller,
                // but do not let it create transaction-local lifecycle state.
                self.graph_in_transaction(name, Some(transaction_id))
            };
            if let Some(graph) = existing {
                return Ok(graph);
            }
        }

        let graph = self.new_detached_graph()?;
        if !self.unframed_writes_allowed() {
            return Ok(graph);
        }
        let mut buffer = self.tx_buffer.write();
        Ok(Arc::clone(
            buffer
                .created_graphs
                .entry(transaction_id)
                .or_default()
                .entry(name.to_string())
                .or_insert(graph),
        ))
    }

    /// Triples visible in `name` (`None` = default graph), including pending ops for `tid`.
    #[must_use]
    pub fn visible_in_graph(&self, name: Option<&str>, tid: Option<TransactionId>) -> Vec<Triple> {
        let pattern = TriplePattern::any();
        match name {
            None => self
                .find_with_pending(&pattern, tid)
                .into_iter()
                .map(|t| (*t).clone())
                .collect(),
            Some(n) => self
                .graph_in_transaction(n, tid)
                .map(|g| {
                    g.find_with_pending(&pattern, tid)
                        .into_iter()
                        .map(|t| (*t).clone())
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// Triples visible in a graph together with their captured valid-time.
    ///
    /// This is the metadata-preserving source for SPARQL COPY/MOVE/ADD. The
    /// transaction's snapshot and pending operations are folded in the same
    /// order as [`visible_in_graph`](Self::visible_in_graph).
    #[doc(hidden)]
    #[must_use]
    pub fn visible_with_valid_in_graph(
        &self,
        name: Option<&str>,
        tid: Option<TransactionId>,
    ) -> Vec<(Triple, Option<ValidTimeInterval>)> {
        match name {
            None => self.visible_with_valid(tid),
            Some(graph) => self
                .graph_in_transaction(graph, tid)
                .map_or_else(Vec::new, |target| target.visible_with_valid(tid)),
        }
    }

    fn visible_with_valid(
        &self,
        tid: Option<TransactionId>,
    ) -> Vec<(Triple, Option<ValidTimeInterval>)> {
        let triples = self.find_with_pending(&TriplePattern::any(), tid);
        let buffer = self.tx_buffer.read();
        let snapshot_epoch = tid.and_then(|tx| buffer.snapshot_epochs.get(&tx).copied());
        let history = self.history.read();
        triples
            .into_iter()
            .map(|triple| {
                let life = history
                    .get(triple.as_ref())
                    .and_then(|versions| match snapshot_epoch {
                        Some(epoch) => versions.iter().find(|life| life.tx.contains(epoch)),
                        None => versions.iter().rev().find(|life| life.tx.is_open()),
                    });
                let mut present = life.is_some();
                let mut valid = life.and_then(|life| life.valid);
                if let Some(transaction_id) = tid
                    && let Some(ops) = buffer.buffers.get(&transaction_id)
                {
                    for op in ops {
                        match op {
                            PendingOp::Delete(pending) if pending.same_identity(&triple) => {
                                present = false;
                                valid = None;
                            }
                            PendingOp::Insert {
                                triple: pending,
                                valid: pending_valid,
                            } if !present && pending.same_identity(&triple) => {
                                present = true;
                                valid = *pending_valid;
                            }
                            _ => {}
                        }
                    }
                }
                ((*triple).clone(), valid)
            })
            .collect()
    }

    fn insert_into_graph_with_valid(
        &self,
        dest: Option<&str>,
        triple: Triple,
        tid: Option<TransactionId>,
        valid: Option<ValidTimeInterval>,
    ) {
        match dest {
            None => {
                if let Some(id) = tid {
                    self.insert_in_transaction_with_valid(id, triple, valid);
                } else {
                    self.insert_at_current_epoch_with_valid(triple, valid);
                }
            }
            Some(n) => {
                let Ok(g) = self.graph_or_create_in_tx(n, tid) else {
                    return;
                };
                if let Some(id) = tid {
                    g.insert_in_transaction_with_valid(id, triple, valid);
                } else {
                    g.insert_at_current_epoch_with_valid(triple, valid);
                }
            }
        }
    }

    fn delete_from_graph(&self, dest: Option<&str>, triple: Triple, tid: Option<TransactionId>) {
        match dest {
            None => {
                if let Some(id) = tid {
                    self.remove_in_transaction(id, triple);
                } else {
                    self.remove(&triple);
                }
            }
            Some(n) => {
                if let Some(id) = tid {
                    if let Some(g) = self.graph_for_mutation_in_tx(n, id) {
                        g.remove_in_transaction(id, triple);
                    }
                } else if let Some(g) = self.graph(n) {
                    g.remove(&triple);
                }
            }
        }
    }

    /// Creates a named graph. Returns `false` if it already exists.
    pub fn create_graph(&self, name: &str) -> bool {
        if !self.unframed_writes_allowed() {
            return false;
        }
        let _lifecycle = self.history_lifecycle_lock.lock();
        let Ok(child) = self.new_detached_graph() else {
            return false;
        };
        let mut graphs = self.named_graphs.write();
        if graphs.contains_key(name) {
            return false;
        }
        graphs.insert(name.to_string(), Arc::clone(&child));
        self.record_graph_create(name, &child, self.current_commit_epoch());
        true
    }

    /// Creates a named graph, staging publication when `tid` is present.
    ///
    /// A graph dropped in this transaction is treated as absent, so
    /// `CREATE GRAPH` after `DROP GRAPH` succeeds (empty graph, RYW).
    pub fn create_graph_in_tx(&self, name: &str, tid: Option<TransactionId>) -> bool {
        if !self.unframed_writes_allowed() {
            return false;
        }
        let Some(transaction_id) = tid else {
            return self.create_graph(name);
        };
        let _lifecycle = self.history_lifecycle_lock.lock();
        let replacing_drop = {
            let buffer = self.tx_buffer.read();
            if buffer
                .created_graphs
                .get(&transaction_id)
                .is_some_and(|graphs| graphs.contains_key(name))
            {
                return false;
            }
            buffer
                .dropped_graphs
                .get(&transaction_id)
                .is_some_and(|graphs| graphs.contains_key(name))
        };
        if !replacing_drop
            && self
                .graph_in_transaction(name, Some(transaction_id))
                .is_some()
        {
            return false;
        }
        let Ok(graph) = self.new_detached_graph() else {
            return false;
        };
        let mut buffer = self.tx_buffer.write();
        let created = buffer.created_graphs.entry(transaction_id).or_default();
        if created.contains_key(name) {
            return false;
        }
        created.insert(name.to_string(), graph);
        true
    }

    /// Drops a named graph. Returns `false` if it didn't exist.
    pub fn drop_graph(&self, name: &str) -> bool {
        if !self.unframed_writes_allowed() {
            return false;
        }
        let _lifecycle = self.history_lifecycle_lock.lock();
        let removed = self.named_graphs.write().remove(name);
        if let Some(graph) = removed {
            self.record_graph_drop(name, &graph, self.current_commit_epoch());
            true
        } else {
            false
        }
    }

    /// Drops a named graph, staging removal when `tid` is present.
    ///
    /// Other sessions continue to see the exact committed partition until
    /// commit. The owner sees the graph as absent immediately.
    pub fn drop_graph_in_tx(&self, name: &str, tid: Option<TransactionId>) -> bool {
        if !self.unframed_writes_allowed() {
            return false;
        }
        match tid {
            None => self.drop_graph(name),
            Some(id) => {
                let mut buffer = self.tx_buffer.write();
                let detached = buffer
                    .created_graphs
                    .get_mut(&id)
                    .and_then(|created| created.remove(name));
                let created_bucket_empty = buffer
                    .created_graphs
                    .get(&id)
                    .is_some_and(HashMap::is_empty);
                if created_bucket_empty {
                    buffer.created_graphs.remove(&id);
                }
                if let Some(detached) = detached {
                    drop(buffer);
                    detached.rollback_dataset(id);
                    return true;
                }
                if buffer
                    .dropped_graphs
                    .get(&id)
                    .is_some_and(|graphs| graphs.contains_key(name))
                {
                    return false;
                }
                let pin = if let Some(touched) = buffer
                    .touched_graphs
                    .get(&id)
                    .and_then(|graphs| graphs.get(name))
                    .cloned()
                {
                    touched
                } else {
                    let Some(store) = self.resolve_named_graph_locked(&mut buffer, name, id) else {
                        return false;
                    };
                    RdfGraphPin {
                        revision: store.lifecycle_revision.load(Ordering::Acquire),
                        store,
                    }
                };
                buffer
                    .dropped_graphs
                    .entry(id)
                    .or_default()
                    .insert(name.to_string(), pin.clone());
                buffer
                    .touched_graphs
                    .entry(id)
                    .or_default()
                    .entry(name.to_string())
                    .or_insert(pin);
                true
            }
        }
    }

    /// Returns all named graph IRIs.
    #[must_use]
    pub fn graph_names(&self) -> Vec<String> {
        self.graph_names_in_transaction(None)
    }

    /// Named graph IRIs visible to one transaction.
    #[must_use]
    pub fn graph_names_in_transaction(&self, transaction_id: Option<TransactionId>) -> Vec<String> {
        let Some(transaction_id) = transaction_id else {
            return self.named_graphs.read().keys().cloned().collect();
        };

        let mut buffer = self.tx_buffer.write();
        let mut names: Vec<String> = if buffer.snapshot_epochs.contains_key(&transaction_id) {
            if !buffer.snapshotted_graph_catalogs.contains(&transaction_id) {
                let committed: Vec<(String, Arc<RdfStore>)> = self
                    .named_graphs
                    .read()
                    .iter()
                    .map(|(name, graph)| (name.clone(), Arc::clone(graph)))
                    .collect();
                let pins = buffer.read_graphs.entry(transaction_id).or_default();
                for (name, graph) in committed {
                    // A prior direct lookup may already have pinned this name
                    // absent; catalog enumeration cannot move that first cut.
                    pins.entry(name).or_insert(Some(graph));
                }
                buffer.snapshotted_graph_catalogs.insert(transaction_id);
            }
            buffer
                .read_graphs
                .get(&transaction_id)
                .into_iter()
                .flat_map(|graphs| graphs.iter())
                .filter(|(_, graph)| graph.is_some())
                .map(|(name, _)| name.clone())
                .collect()
        } else {
            self.named_graphs.read().keys().cloned().collect()
        };
        if let Some(dropped) = buffer.dropped_graphs.get(&transaction_id) {
            names.retain(|name| !dropped.contains_key(name));
        }
        if let Some(created) = buffer.created_graphs.get(&transaction_id) {
            for name in created.keys() {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
        }
        names
    }

    /// Returns the number of named graphs.
    #[must_use]
    pub fn graph_count(&self) -> usize {
        self.named_graphs.read().len()
    }

    /// Clears a specific graph, or the default graph if `name` is `None`.
    pub fn clear_graph(&self, name: Option<&str>) {
        if !self.unframed_writes_allowed() {
            return;
        }
        self.clear_graph_in_tx(name, None);
    }

    /// Clears a graph, buffering deletes when `tid` is set.
    ///
    /// `Some("")` means CLEAR ALL (default + every named graph).
    pub fn clear_graph_in_tx(&self, name: Option<&str>, tid: Option<TransactionId>) {
        if !self.unframed_writes_allowed() {
            return;
        }
        if name == Some("") {
            self.clear_graph_in_tx(None, tid);
            for n in self.graph_names_in_transaction(tid) {
                self.clear_graph_in_tx(Some(&n), tid);
            }
            return;
        }
        if tid.is_none() {
            match name {
                None => self.clear(),
                Some(n) => {
                    if let Some(g) = self.named_graphs.read().get(n) {
                        g.clear();
                    }
                }
            }
            return;
        }
        for t in self.visible_in_graph(name, tid) {
            self.delete_from_graph(name, t, tid);
        }
    }

    /// Clears all named graphs (but not the default graph).
    pub fn clear_all_named(&self) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let _lifecycle = self.history_lifecycle_lock.lock();
        let removed = std::mem::take(&mut *self.named_graphs.write());
        let epoch = self.current_commit_epoch();
        for (name, graph) in removed {
            self.record_graph_drop(&name, &graph, epoch);
        }
    }

    /// Copies all triples from source graph to destination graph.
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) {
        if !self.unframed_writes_allowed() {
            return;
        }
        self.copy_graph_in_tx(source, dest, None);
    }

    /// COPY, buffering when `tid` is set so rollback restores dest.
    pub fn copy_graph_in_tx(
        &self,
        source: Option<&str>,
        dest: Option<&str>,
        tid: Option<TransactionId>,
    ) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let triples = self.visible_with_valid_in_graph(source, tid);
        self.clear_graph_in_tx(dest, tid);
        for (triple, valid) in triples {
            self.insert_into_graph_with_valid(dest, triple, tid, valid);
        }
    }

    /// Moves all triples from source graph to destination graph.
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn move_graph(&self, source: Option<&str>, dest: Option<&str>) {
        if !self.unframed_writes_allowed() {
            return;
        }
        self.move_graph_in_tx(source, dest, None);
    }

    /// MOVE, buffering when `tid` is set. Named source is dropped on commit.
    pub fn move_graph_in_tx(
        &self,
        source: Option<&str>,
        dest: Option<&str>,
        tid: Option<TransactionId>,
    ) {
        if !self.unframed_writes_allowed() {
            return;
        }
        if source == dest {
            return;
        }
        self.copy_graph_in_tx(source, dest, tid);
        match (source, tid) {
            (Some(n), Some(id)) => {
                let _ = self.drop_graph_in_tx(n, Some(id));
            }
            (Some(n), None) => {
                let _ = self.drop_graph(n);
            }
            (None, _) => self.clear_graph_in_tx(None, tid),
        }
    }

    /// Adds all triples from source graph into destination graph (union).
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn add_graph(&self, source: Option<&str>, dest: Option<&str>) {
        if !self.unframed_writes_allowed() {
            return;
        }
        self.add_graph_in_tx(source, dest, None);
    }

    /// ADD, buffering when `tid` is set so rollback restores dest.
    pub fn add_graph_in_tx(
        &self,
        source: Option<&str>,
        dest: Option<&str>,
        tid: Option<TransactionId>,
    ) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let mut present: FxHashSet<_> = self
            .visible_in_graph(dest, tid)
            .iter()
            .map(Triple::canonical_identity_key)
            .collect();
        let triples = self.visible_with_valid_in_graph(source, tid);
        for (triple, valid) in triples {
            if present.insert(triple.canonical_identity_key()) {
                self.insert_into_graph_with_valid(dest, triple, tid, valid);
            }
        }
    }

    /// Finds triples across specific graphs.
    ///
    /// - `graphs = None` searches the default graph only (backward compatible).
    /// - `graphs = Some(&[])` searches all named graphs (excluding the default graph).
    /// - `graphs = Some(&["g1", "g2"])` searches those named graphs only.
    pub fn find_in_graphs(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
    ) -> Vec<(Option<String>, Arc<Triple>)> {
        self.find_in_graphs_with_pending(pattern, graphs, None)
    }

    /// [`Self::find_in_graphs`] plus the caller's uncommitted SPARQL writes.
    pub fn find_in_graphs_with_pending(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
        transaction_id: Option<TransactionId>,
    ) -> Vec<(Option<String>, Arc<Triple>)> {
        match graphs {
            None => {
                // Default graph only
                self.find_with_pending(pattern, transaction_id)
                    .into_iter()
                    .map(|t| (None, t))
                    .collect()
            }
            Some([]) => {
                // All named graphs (excludes default graph per SPARQL spec sec 13.3)
                let mut results = Vec::new();
                for name in self.graph_names_in_transaction(transaction_id) {
                    if let Some(store) = self.graph_in_transaction(&name, transaction_id) {
                        for t in store.find_with_pending(pattern, transaction_id) {
                            results.push((Some(name.clone()), t));
                        }
                    }
                }
                results
            }
            Some(names) => {
                // Specific named graphs
                let mut results = Vec::new();
                for name in names {
                    if let Some(store) = self.graph_in_transaction(name, transaction_id) {
                        for t in store.find_with_pending(pattern, transaction_id) {
                            results.push((Some((*name).to_string()), t));
                        }
                    }
                }
                results
            }
        }
    }

    // =========================================================================
    // Transaction support
    // =========================================================================

    /// Registers the committed epoch visible to a snapshot-isolated transaction.
    ///
    /// The first epoch wins, so repeated query planning cannot move a live
    /// transaction's snapshot forward. Existing named graphs receive the same
    /// cut; graphs discovered or created later inherit it at the graph boundary.
    #[doc(hidden)]
    pub fn register_transaction_snapshot(&self, transaction_id: TransactionId, epoch: EpochId) {
        if !self.unframed_writes_allowed() {
            return;
        }
        {
            let mut buffer = self.tx_buffer.write();
            buffer
                .snapshot_epochs
                .entry(transaction_id)
                .or_insert(epoch);
            buffer
                .write_revisions
                .entry(transaction_id)
                .or_insert_with(|| self.lifecycle_revision.load(Ordering::Acquire));
        }
        let graphs: Vec<Arc<RdfStore>> = self.named_graphs.read().values().cloned().collect();
        for graph in graphs {
            graph.register_transaction_snapshot(transaction_id, epoch);
        }
        let detached: Vec<Arc<RdfStore>> = self
            .tx_buffer
            .read()
            .created_graphs
            .get(&transaction_id)
            .into_iter()
            .flat_map(|graphs| graphs.values().cloned())
            .collect();
        for graph in detached {
            graph.register_transaction_snapshot(transaction_id, epoch);
        }
    }

    /// Inserts a triple within a transaction context.
    ///
    /// The insert is buffered until the transaction is committed.
    /// If the transaction is rolled back, the insert is discarded.
    pub fn insert_in_transaction(&self, transaction_id: TransactionId, triple: Triple) {
        self.insert_in_transaction_with_valid(transaction_id, triple, None);
    }

    /// Inserts a triple with transaction-local application valid-time metadata.
    ///
    /// Capturing validity in the pending operation keeps concurrent transactions
    /// independent and makes rollback/error paths automatically discard it.
    #[doc(hidden)]
    pub fn insert_in_transaction_with_valid(
        &self,
        transaction_id: TransactionId,
        triple: Triple,
        valid: Option<ValidTimeInterval>,
    ) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let mut buffer = self.tx_buffer.write();
        buffer
            .write_revisions
            .entry(transaction_id)
            .or_insert_with(|| self.lifecycle_revision.load(Ordering::Acquire));
        buffer
            .buffers
            .entry(transaction_id)
            .or_default()
            .push(PendingOp::Insert { triple, valid });
    }

    /// Removes a triple within a transaction context.
    ///
    /// The removal is buffered until the transaction is committed.
    /// If the transaction is rolled back, the removal is discarded.
    pub fn remove_in_transaction(&self, transaction_id: TransactionId, triple: Triple) {
        if !self.unframed_writes_allowed() {
            return;
        }
        let mut buffer = self.tx_buffer.write();
        buffer
            .write_revisions
            .entry(transaction_id)
            .or_insert_with(|| self.lifecycle_revision.load(Ordering::Acquire));
        buffer
            .buffers
            .entry(transaction_id)
            .or_default()
            .push(PendingOp::Delete(triple));
    }

    /// Commits the dataset and stamps RDF transaction-time with a real `epoch`.
    ///
    /// # Errors
    ///
    /// Returns an error before consuming buffered operations if `epoch` is
    /// [`EpochId::PENDING`] or this thread lacks authority for a sealed store.
    pub fn try_commit_dataset_at(
        &self,
        transaction_id: TransactionId,
        epoch: EpochId,
    ) -> grafeo_common::utils::error::Result<usize> {
        if epoch == EpochId::PENDING {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "RDF transaction commit epoch cannot be PENDING".to_string(),
            ));
        }
        let _commit = self.commit_lock.lock();
        self.try_commit_dataset_at_under_gate(transaction_id, epoch)
    }

    /// Commits while the caller already holds [`Self::lock_commit`].
    ///
    /// # Errors
    ///
    /// Returns an error before consuming buffered operations if `epoch` is
    /// [`EpochId::PENDING`] or this thread lacks authority for a sealed store.
    #[doc(hidden)]
    pub fn try_commit_dataset_at_under_gate(
        &self,
        transaction_id: TransactionId,
        epoch: EpochId,
    ) -> grafeo_common::utils::error::Result<usize> {
        self.try_set_commit_epoch(epoch)?;
        Ok(self.commit_dataset_under_gate(transaction_id))
    }

    /// Commits a transaction, applying all buffered operations.
    ///
    /// Returns the number of operations applied.
    pub fn commit_transaction(&self, transaction_id: TransactionId) -> usize {
        if !self.unframed_writes_allowed() {
            return 0;
        }
        let ops = {
            let mut buffer = self.tx_buffer.write();
            buffer.snapshot_epochs.remove(&transaction_id);
            buffer.write_revisions.remove(&transaction_id);
            buffer.read_graphs.remove(&transaction_id);
            buffer.snapshotted_graph_catalogs.remove(&transaction_id);
            buffer.buffers.remove(&transaction_id).unwrap_or_default()
        };

        let count = ops.len();
        for op in ops {
            match op {
                PendingOp::Insert { triple, valid } => {
                    self.insert_at_current_epoch_with_valid(triple, valid);
                }
                PendingOp::Delete(triple) => {
                    self.remove(&triple);
                }
            }
        }
        count
    }

    fn validate_write_revision(&self, transaction_id: TransactionId) -> Result<(), String> {
        let buffer = self.tx_buffer.read();
        if buffer
            .buffers
            .get(&transaction_id)
            .is_some_and(|ops| !ops.is_empty())
            && buffer
                .write_revisions
                .get(&transaction_id)
                .is_some_and(|revision| {
                    *revision != self.lifecycle_revision.load(Ordering::Acquire)
                })
        {
            return Err("RDF graph changed concurrently; transaction cannot commit its statement representatives".to_string());
        }
        Ok(())
    }

    /// Validates transaction-local named-graph lifecycle pins.
    ///
    /// Callers must run this while holding their database publication gate and
    /// before making the transaction durable. A successful validation means
    /// every shared partition written or dropped is still the exact `Arc` at
    /// the revision observed on first touch, and every detached create still
    /// targets an absent name (unless it replaces its pinned drop).
    ///
    /// # Errors
    ///
    /// Returns an error if a written RDF partition changed after its first
    /// snapshot/write, or a named graph was concurrently created, removed or
    /// replaced. Partition revisions conservatively reject unrelated writes in
    /// the same graph, matching the named-graph lifecycle conflict boundary.
    pub fn validate_transaction_lifecycle(
        &self,
        transaction_id: TransactionId,
    ) -> Result<(), String> {
        self.validate_write_revision(transaction_id)?;
        let (created, dropped, touched) = {
            let buffer = self.tx_buffer.read();
            (
                buffer
                    .created_graphs
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default(),
                buffer
                    .dropped_graphs
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default(),
                buffer
                    .touched_graphs
                    .get(&transaction_id)
                    .cloned()
                    .unwrap_or_default(),
            )
        };
        let graphs = self.named_graphs.read();

        for (name, pin) in &touched {
            let Some(current) = graphs.get(name) else {
                return Err(format!(
                    "RDF named graph <{name}> was removed concurrently; transaction cannot commit"
                ));
            };
            if !Arc::ptr_eq(current, &pin.store) {
                return Err(format!(
                    "RDF named graph <{name}> was replaced concurrently; transaction cannot commit"
                ));
            }
            current.validate_write_revision(transaction_id)?;
            if current.lifecycle_revision.load(Ordering::Acquire) != pin.revision {
                return Err(format!(
                    "RDF named graph <{name}> changed concurrently; transaction cannot commit"
                ));
            }
        }

        for name in created.keys() {
            if dropped.contains_key(name) {
                continue;
            }
            if graphs.contains_key(name) {
                return Err(format!(
                    "RDF named graph <{name}> was created concurrently; transaction cannot commit"
                ));
            }
        }
        Ok(())
    }

    /// Commits buffered RDF ops on the default graph and every named graph.
    ///
    /// Named-graph `INSERT DATA { GRAPH … }` buffers on the per-graph store.
    /// Committing only the default graph dropped those writes.
    ///
    /// # Panics
    ///
    /// Panics if a named graph disappears after lifecycle validation while the
    /// caller is publishing the transaction. Correct callers hold the database
    /// publication gate across validation and this method.
    pub fn commit_dataset(&self, transaction_id: TransactionId) -> usize {
        if !self.unframed_writes_allowed() {
            return 0;
        }
        let _commit = self.commit_lock.lock();
        self.commit_dataset_under_gate(transaction_id)
    }

    /// Commits while the caller already holds [`Self::lock_commit`].
    #[doc(hidden)]
    pub fn commit_dataset_under_gate(&self, transaction_id: TransactionId) -> usize {
        if !self.unframed_writes_allowed() {
            return 0;
        }

        let (created, dropped) = {
            let mut buffer = self.tx_buffer.write();
            buffer.touched_graphs.remove(&transaction_id);
            (
                buffer
                    .created_graphs
                    .remove(&transaction_id)
                    .unwrap_or_default(),
                buffer
                    .dropped_graphs
                    .remove(&transaction_id)
                    .unwrap_or_default(),
            )
        };

        let mut n = self.commit_transaction(transaction_id);
        let shared: Vec<(String, Arc<RdfStore>)> = self
            .named_graphs
            .read()
            .iter()
            .map(|(name, graph)| (name.clone(), Arc::clone(graph)))
            .collect();
        for (name, graph) in &shared {
            if dropped
                .get(name)
                .is_some_and(|pin| Arc::ptr_eq(&pin.store, graph))
            {
                n += graph.rollback_transaction(transaction_id);
            } else {
                n += graph.commit_transaction(transaction_id);
            }
        }
        for graph in created.values() {
            n += graph.commit_transaction(transaction_id);
        }

        // Validation happened before the durable marker while the engine held
        // its publication gate. Re-check exact identities while holding the
        // registry lock: violating this invariant is fail-stop, allowing WAL
        // recovery to reconstruct the committed result instead of publishing a
        // partial lifecycle change.
        let _lifecycle = self.history_lifecycle_lock.lock();
        let mut graphs = self.named_graphs.write();
        for (name, pin) in &dropped {
            let current = graphs.get(name).unwrap_or_else(|| {
                panic!("validated RDF graph <{name}> disappeared before publication")
            });
            assert!(
                Arc::ptr_eq(current, &pin.store)
                    && current.lifecycle_revision.load(Ordering::Acquire) == pin.revision,
                "validated RDF graph <{name}> changed before publication"
            );
        }
        for name in created.keys().filter(|name| !dropped.contains_key(*name)) {
            assert!(
                !graphs.contains_key(name),
                "validated RDF graph <{name}> appeared before publication"
            );
        }
        let epoch = self.current_commit_epoch();
        for (name, pin) in &dropped {
            self.record_graph_drop(name, &pin.store, epoch);
            if let Some(replacement) = created.get(name) {
                graphs.insert(name.clone(), Arc::clone(replacement));
                self.record_graph_create(name, replacement, epoch);
            } else if graphs
                .get(name)
                .is_some_and(|current| Arc::ptr_eq(current, &pin.store))
            {
                graphs.remove(name);
            }
        }
        for (name, graph) in created
            .into_iter()
            .filter(|(name, _)| !dropped.contains_key(name))
        {
            graphs.insert(name.clone(), Arc::clone(&graph));
            self.record_graph_create(&name, &graph, epoch);
        }
        n
    }

    /// Rolls back a transaction, discarding all buffered operations.
    ///
    /// Returns the number of operations discarded.
    pub fn rollback_transaction(&self, transaction_id: TransactionId) -> usize {
        if !self.unframed_writes_allowed() {
            return 0;
        }
        let mut buffer = self.tx_buffer.write();
        buffer.snapshot_epochs.remove(&transaction_id);
        buffer.write_revisions.remove(&transaction_id);
        buffer.read_graphs.remove(&transaction_id);
        buffer.snapshotted_graph_catalogs.remove(&transaction_id);
        buffer
            .buffers
            .remove(&transaction_id)
            .map_or(0, |ops| ops.len())
    }

    /// Rolls back buffered RDF ops on the default graph and every named graph.
    pub fn rollback_dataset(&self, transaction_id: TransactionId) -> usize {
        if !self.unframed_writes_allowed() {
            return 0;
        }

        let mut partitions = Vec::new();
        let mut seen = FxHashSet::default();
        for graph in self.named_graphs.read().values() {
            Self::retain_exact_partition(&mut partitions, &mut seen, graph);
        }
        {
            let mut buffer = self.tx_buffer.write();
            let created = buffer
                .created_graphs
                .remove(&transaction_id)
                .unwrap_or_default();
            let dropped = buffer
                .dropped_graphs
                .remove(&transaction_id)
                .unwrap_or_default();
            let touched = buffer
                .touched_graphs
                .remove(&transaction_id)
                .unwrap_or_default();
            for graph in created.values() {
                Self::retain_exact_partition(&mut partitions, &mut seen, graph);
            }
            for pin in dropped.values() {
                Self::retain_exact_partition(&mut partitions, &mut seen, &pin.store);
            }
            for pin in touched.values() {
                Self::retain_exact_partition(&mut partitions, &mut seen, &pin.store);
            }
        }
        let mut n = self.rollback_transaction(transaction_id);
        for graph in partitions {
            n += graph.rollback_transaction(transaction_id);
        }
        n
    }

    /// Checks if a transaction has pending operations.
    #[must_use]
    pub fn has_pending_ops(&self, transaction_id: TransactionId) -> bool {
        let buffer = self.tx_buffer.read();
        buffer
            .buffers
            .get(&transaction_id)
            .is_some_and(|ops| !ops.is_empty())
    }

    /// Returns triples matching the given pattern, including pending inserts
    /// and excluding pending deletes from the specified transaction
    /// (for read-your-writes within a transaction).
    ///
    /// This provides snapshot isolation semantics: within a transaction, you see
    /// the committed start-epoch cut plus all your own pending changes (inserts
    /// and deletes) as if they were committed. Transactions without a registered
    /// cut (including Read Committed) read the live indexes.
    pub fn find_with_pending(
        &self,
        pattern: &TriplePattern,
        transaction_id: Option<TransactionId>,
    ) -> Vec<Arc<Triple>> {
        let snapshot_epoch =
            transaction_id.and_then(|tx| self.tx_buffer.read().snapshot_epochs.get(&tx).copied());
        let mut results = snapshot_epoch.map_or_else(
            || self.find(pattern),
            |epoch| self.find_at_epoch(pattern, epoch),
        );

        if let Some(tx) = transaction_id {
            let buffer = self.tx_buffer.read();
            if let Some(ops) = buffer.buffers.get(&tx) {
                for op in ops {
                    match op {
                        PendingOp::Delete(triple) => {
                            results.retain(|t| !t.same_identity(triple));
                        }
                        PendingOp::Insert { triple, .. } if pattern.matches(triple) => {
                            if !results.iter().any(|current| current.same_identity(triple)) {
                                results.push(Arc::new(triple.clone()));
                            }
                        }
                        PendingOp::Insert { .. } => {}
                    }
                }
            }
        }

        results
    }
}

impl Default for RdfStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about an RDF store.
#[derive(Debug, Clone, Copy)]
pub struct RdfStoreStats {
    /// Total number of triples.
    pub triple_count: usize,
    /// Number of unique subjects.
    pub subject_count: usize,
    /// Number of unique predicates.
    pub predicate_count: usize,
    /// Number of unique objects (0 if object index disabled).
    pub object_count: usize,
    /// Number of named graphs.
    pub graph_count: usize,
}

/// Result of a bulk load operation.
#[derive(Debug, Clone)]
pub struct BulkLoadResult {
    /// Number of triples loaded.
    pub triple_count: usize,
    /// Statistics computed during the load pass.
    pub statistics: crate::statistics::RdfStatistics,
}

/// Error from parsing an N-Triples document.
#[derive(Debug)]
#[non_exhaustive]
pub enum NTriplesError {
    /// I/O error while reading.
    Io(std::io::Error),
    /// A line could not be parsed as a valid N-Triples triple.
    Parse {
        /// 1-based line number.
        line: usize,
        /// The raw line content.
        content: String,
    },
}

impl std::fmt::Display for NTriplesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Parse { line, content } => {
                write!(f, "parse error at line {line}: {content}")
            }
        }
    }
}

impl std::error::Error for NTriplesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Parse { .. } => None,
        }
    }
}

/// Extracts the next N-Triples term from a string, returning `(term_str, rest)`.
fn next_ntriples_term(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('<') {
        // IRI: find closing >
        let end = rest.find('>')?;
        Some((&s[..end + 2], &rest[end + 1..]))
    } else if s.starts_with("_:") {
        // Blank node: until whitespace or end
        let end = s.find(|c: char| c.is_whitespace()).unwrap_or(s.len());
        Some((&s[..end], &s[end..]))
    } else if s.starts_with('"') {
        // Literal: find closing quote (handling escapes), then optional suffix
        let bytes = s.as_bytes();
        let mut pos = 1;
        while pos < bytes.len() {
            if bytes[pos] == b'\\' {
                pos += 2; // skip escape sequence
            } else if bytes[pos] == b'"' {
                pos += 1;
                // Check for datatype or language suffix
                if s[pos..].starts_with("^^<") {
                    if let Some(end) = s[pos..].find('>') {
                        pos += end + 1;
                    }
                } else if s[pos..].starts_with('@') {
                    let lang_end = s[pos..]
                        .find(|c: char| c.is_whitespace())
                        .unwrap_or(s.len() - pos);
                    pos += lang_end;
                }
                break;
            } else {
                pos += 1;
            }
        }
        Some((&s[..pos], &s[pos..]))
    } else {
        None
    }
}

/// Parses a single N-Triples line into a `Triple`.
///
/// Expected format: `<subject> <predicate> <object> .`
fn parse_ntriples_line(line: &str) -> Option<Triple> {
    let (subj_str, rest) = next_ntriples_term(line)?;
    let (pred_str, rest) = next_ntriples_term(rest)?;
    let (obj_str, rest) = next_ntriples_term(rest)?;

    // Expect trailing ` .`
    let rest = rest.trim();
    if !rest.starts_with('.') {
        return None;
    }

    let subject = Term::from_ntriples(subj_str)?;
    let predicate = Term::from_ntriples(pred_str)?;
    let object = Term::from_ntriples(obj_str)?;
    Some(Triple::new(subject, predicate, object))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_dataset_identity_propagates_to_named_graphs() {
        let store_id = StoreId::from_bytes([0x5a; StoreId::LEN]).unwrap();
        let store = RdfStore::with_config_and_store_id(RdfStoreConfig::default(), store_id);

        assert_eq!(store.store_id(), store_id);
        assert!(store.create_graph("http://example.org/named"));
        assert_eq!(
            store.graph("http://example.org/named").unwrap().store_id(),
            store_id
        );
    }

    #[test]
    fn pending_commit_epoch_is_rejected_before_any_dataset_mutation() {
        use grafeo_common::utils::error::ErrorCode;

        let store = RdfStore::new();
        let live_name = "http://example.org/live";
        let detached_name = "http://example.org/detached";
        let lifecycle_transaction = TransactionId::new(70);
        assert!(store.create_graph(live_name));
        let live = store.graph(live_name).unwrap();
        let detached = store
            .graph_or_create_in_tx(detached_name, Some(lifecycle_transaction))
            .unwrap();
        store.try_set_commit_epoch(EpochId::new(7)).unwrap();

        let transaction_id = TransactionId::new(71);
        let triple = sample_triples()[0].clone();
        store.insert_in_transaction(transaction_id, triple.clone());
        assert!(store.has_pending_ops(transaction_id));

        let error = store
            .try_commit_dataset_at(transaction_id, EpochId::PENDING)
            .unwrap_err();
        assert_eq!(error.error_code(), ErrorCode::InvalidInput);
        assert_eq!(store.commit_epoch(), EpochId::new(7));
        assert_eq!(live.commit_epoch(), EpochId::new(7));
        assert_eq!(detached.commit_epoch(), EpochId::new(7));
        assert!(store.has_pending_ops(transaction_id));
        assert!(!store.contains(&triple));

        let error = store.try_set_commit_epoch(EpochId::PENDING).unwrap_err();
        assert_eq!(error.error_code(), ErrorCode::InvalidInput);
        assert_eq!(store.commit_epoch(), EpochId::new(7));
        assert_eq!(live.commit_epoch(), EpochId::new(7));
        assert_eq!(detached.commit_epoch(), EpochId::new(7));

        let zero = RdfStore::new();
        let zero_transaction = TransactionId::new(72);
        let zero_triple = sample_triples()[1].clone();
        zero.insert_in_transaction(zero_transaction, zero_triple.clone());
        assert_eq!(
            zero.try_commit_dataset_at(zero_transaction, EpochId::new(0))
                .unwrap(),
            1
        );
        assert_eq!(zero.commit_epoch(), EpochId::new(0));
        assert!(
            zero.triples_at_epoch(EpochId::new(0))
                .iter()
                .any(|candidate| candidate.as_ref() == &zero_triple)
        );
    }

    #[test]
    fn explicit_pending_epoch_mutators_reject_before_live_or_history_change() {
        use grafeo_common::utils::error::ErrorCode;

        let store = RdfStore::new();
        store.try_set_commit_epoch(EpochId::new(4)).unwrap();
        let existing = sample_triples()[0].clone();
        let candidate = sample_triples()[1].clone();
        assert!(
            store
                .try_insert_at_epoch_with_valid(existing.clone(), EpochId::new(4), None)
                .unwrap()
        );
        let before = store.dataset_history().unwrap();

        let error = store
            .try_insert_at_epoch_with_valid(candidate.clone(), EpochId::PENDING, None)
            .unwrap_err();
        assert_eq!(error.error_code(), ErrorCode::InvalidInput);
        assert!(!store.contains(&candidate));

        let error = store
            .try_remove_at_epoch(&existing, EpochId::PENDING)
            .unwrap_err();
        assert_eq!(error.error_code(), ErrorCode::InvalidInput);
        assert!(store.contains(&existing));

        let after = store.dataset_history().unwrap();
        assert_eq!(store.commit_epoch(), EpochId::new(4));
        assert_eq!(after.store_id(), before.store_id());
        assert_eq!(after.completeness(), before.completeness());
        assert_eq!(
            after.next_graph_incarnation(),
            before.next_graph_incarnation()
        );
        assert_eq!(after.graph_lives(), before.graph_lives());
        assert_eq!(after.quad_versions(), before.quad_versions());

        let zero = RdfStore::new();
        assert!(
            zero.try_insert_at_epoch_with_valid(candidate.clone(), EpochId::INITIAL, None)
                .unwrap()
        );
        assert!(zero.contains(&candidate));
        assert!(
            zero.try_remove_at_epoch(&candidate, EpochId::INITIAL)
                .unwrap()
        );
        assert!(!zero.contains(&candidate));
        assert!(zero.dataset_history().unwrap().quad_versions().is_empty());
    }

    #[test]
    fn pending_or_incoherent_dataset_history_cut_cannot_replace_live_state() {
        let target = RdfStore::new();
        target.try_set_commit_epoch(EpochId::new(7)).unwrap();
        let existing = sample_triples()[0].clone();
        assert!(target.insert(existing));
        assert!(target.create_graph("http://example.org/preserved"));
        target.remember_projection(91, "http://example.org/Type", "Type");

        let assert_unchanged = |before: &RdfDatasetHistory| {
            let after = target.dataset_history().unwrap();
            assert_eq!(target.commit_epoch(), EpochId::new(7));
            assert_eq!(after.store_id(), before.store_id());
            assert_eq!(after.completeness(), before.completeness());
            assert_eq!(
                after.next_graph_incarnation(),
                before.next_graph_incarnation()
            );
            assert_eq!(after.graph_lives(), before.graph_lives());
            assert_eq!(after.quad_versions(), before.quad_versions());
            assert_eq!(
                target.projection(91),
                Some((
                    91,
                    "http://example.org/Type".to_string(),
                    "Type".to_string()
                ))
            );
        };
        let before = target.dataset_history().unwrap();
        let empty = RdfDatasetHistory::new(
            target.store_id(),
            HistoryCompleteness::Complete,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        let error = target
            .replace_dataset_history_exact(empty.clone(), EpochId::PENDING)
            .unwrap_err();
        assert!(error.contains("cannot be PENDING"), "{error}");
        assert_unchanged(&before);

        let future = RdfStore::new();
        future.try_set_commit_epoch(EpochId::new(9)).unwrap();
        assert!(future.insert(sample_triples()[1].clone()));
        let error = target
            .replace_dataset_history_exact(future.dataset_history().unwrap(), EpochId::new(8))
            .unwrap_err();
        assert!(error.contains("beyond its commit epoch"), "{error}");
        assert_unchanged(&before);
    }

    fn sample_triples() -> Vec<Triple> {
        vec![
            Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/age"),
                Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
            ),
            Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/gus"),
            ),
            Triple::new(
                Term::iri("http://example.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
        ]
    }

    fn labelled_triple(label: &str) -> Triple {
        Triple::new(
            Term::iri(format!("http://example.org/{label}")),
            Term::iri("http://example.org/value"),
            Term::literal(label),
        )
    }

    #[test]
    fn sealed_store_rejects_raw_logical_mutators() {
        use crate::graph::write_permit::{WriteAuthority, with_authority};

        let store = RdfStore::new();
        let existing = sample_triples()[0].clone();
        let candidate = Triple::new(
            Term::iri("http://example.org/candidate"),
            Term::iri("http://example.org/p"),
            Term::literal("candidate"),
        );
        assert!(store.insert(existing.clone()));
        assert!(store.create_graph("http://example.org/existing-graph"));
        let named = store.graph("http://example.org/existing-graph").unwrap();
        assert!(named.insert(existing.clone()));
        store.remember_projection(1, "http://example.org/Type", "Type");
        store.try_set_commit_epoch(EpochId::new(3)).unwrap();

        let owner = WriteAuthority::new();
        let wrong = WriteAuthority::new();
        assert!(store.seal_unframed_writes(&owner));

        assert!(!store.insert(candidate.clone()));
        assert!(!store.insert_with_valid(candidate.clone(), 10, 20));
        assert!(
            store
                .try_insert_at_epoch_with_valid(
                    candidate.clone(),
                    EpochId::new(9),
                    Some(ValidTimeInterval::from_tai_nanoseconds(10, 20).unwrap()),
                )
                .is_err()
        );
        assert_eq!(store.batch_insert([candidate.clone()]), 0);
        assert!(!store.remove(&existing));
        assert!(
            store
                .try_remove_at_epoch(&existing, EpochId::new(9))
                .is_err()
        );
        store.clear();
        store.restore_quad_version(
            candidate.clone(),
            QuadLife {
                tx: EpochInterval::open(EpochId::new(9)),
                valid: Some(ValidTimeInterval::from_tai_nanoseconds(10, 20).unwrap()),
            },
        );
        assert!(store.try_set_commit_epoch(EpochId::new(99)).is_err());
        store.remember_projection(2, "http://example.org/Other", "Other");

        assert!(store.contains(&existing));
        assert!(!store.contains(&candidate));
        assert_eq!(store.commit_epoch(), EpochId::new(3));
        assert!(store.projection(1).is_some());
        assert!(store.projection(2).is_none());
        assert!(
            store
                .quad_history()
                .iter()
                .all(|(triple, _)| triple.as_ref() != &candidate)
        );

        assert_eq!(store.bulk_load([candidate.clone()]).triple_count, 0);
        let ntriples = b"<http://example.org/load> <http://example.org/p> \"load\" .\n".as_slice();
        assert_eq!(
            store
                .load_ntriples(std::io::Cursor::new(ntriples))
                .unwrap()
                .triple_count,
            0
        );
        assert_eq!(
            store
                .load_ntriples_streaming(std::io::Cursor::new(ntriples), 1)
                .unwrap(),
            0
        );
        let turtle = "@prefix ex: <http://example.org/> . ex:load ex:p \"load\" .";
        assert_eq!(store.load_turtle(turtle).unwrap().triple_count, 0);
        assert_eq!(store.load_turtle_streaming(turtle, 1).unwrap(), 0);
        assert_eq!(
            store
                .load_turtle_reader(std::io::Cursor::new(turtle), 1)
                .unwrap(),
            0
        );
        assert_eq!(store.len(), 1);

        assert!(!store.create_graph("http://example.org/raw-create"));
        assert!(!store.create_graph_in_tx(
            "http://example.org/raw-create-tx",
            Some(TransactionId::new(40)),
        ));
        let detached = store
            .graph_or_create("http://example.org/raw-or-create")
            .unwrap();
        assert!(store.graph("http://example.org/raw-or-create").is_none());
        assert!(!detached.insert(candidate.clone()));
        assert!(!store.drop_graph("http://example.org/existing-graph"));
        assert!(!store.drop_graph_in_tx(
            "http://example.org/existing-graph",
            Some(TransactionId::new(41)),
        ));
        store.clear_graph(None);
        store.clear_graph_in_tx(None, Some(TransactionId::new(42)));
        store.clear_all_named();
        store.copy_graph(None, Some("http://example.org/raw-copy"));
        store.copy_graph_in_tx(
            None,
            Some("http://example.org/raw-copy-tx"),
            Some(TransactionId::new(43)),
        );
        store.move_graph(
            Some("http://example.org/existing-graph"),
            Some("http://example.org/raw-move"),
        );
        store.move_graph_in_tx(
            Some("http://example.org/existing-graph"),
            Some("http://example.org/raw-move-tx"),
            Some(TransactionId::new(44)),
        );
        store.add_graph(None, Some("http://example.org/raw-add"));
        store.add_graph_in_tx(
            None,
            Some("http://example.org/raw-add-tx"),
            Some(TransactionId::new(45)),
        );
        assert_eq!(store.graph_count(), 1);
        assert_eq!(named.len(), 1);

        let mutation_target_tid = TransactionId::new(46);
        assert!(
            store
                .graph_for_mutation_in_tx("http://example.org/existing-graph", mutation_target_tid,)
                .is_none()
        );
        with_authority(&wrong, || {
            assert!(
                store
                    .graph_for_mutation_in_tx(
                        "http://example.org/existing-graph",
                        mutation_target_tid,
                    )
                    .is_none()
            );
        });
        let raw_existing = store
            .graph_or_create_in_tx(
                "http://example.org/existing-graph",
                Some(mutation_target_tid),
            )
            .expect("sealed raw lookup preserves the existing read result");
        assert!(Arc::ptr_eq(&raw_existing, &named));
        assert!(
            !store
                .tx_buffer
                .read()
                .touched_graphs
                .contains_key(&mutation_target_tid),
            "foreign callers must not create lifecycle pins"
        );
        let authorized_target = with_authority(&owner, || {
            store
                .graph_for_mutation_in_tx("http://example.org/existing-graph", mutation_target_tid)
                .expect("owner authority resolves the mutation target")
        });
        assert!(Arc::ptr_eq(&authorized_target, &named));
        with_authority(&owner, || {
            assert_eq!(store.rollback_dataset(mutation_target_tid), 0);
        });

        let raw_tid = TransactionId::new(50);
        store.insert_in_transaction(raw_tid, candidate.clone());
        store.remove_in_transaction(raw_tid, existing.clone());
        assert!(!store.has_pending_ops(raw_tid));

        let commit_tid = TransactionId::new(51);
        with_authority(&owner, || {
            store.insert_in_transaction_with_valid(
                commit_tid,
                candidate.clone(),
                Some(ValidTimeInterval::from_tai_nanoseconds(100, 200).unwrap()),
            );
        });
        assert!(store.has_pending_ops(commit_tid));
        assert!(
            store
                .try_commit_dataset_at(commit_tid, EpochId::new(10))
                .is_err()
        );
        assert_eq!(store.commit_epoch(), EpochId::new(3));
        assert!(store.has_pending_ops(commit_tid));
        with_authority(&wrong, || {
            assert!(
                store
                    .try_commit_dataset_at(commit_tid, EpochId::new(10))
                    .is_err()
            );
        });
        assert!(store.has_pending_ops(commit_tid));
        assert_eq!(
            with_authority(&owner, || {
                store
                    .try_commit_dataset_at(commit_tid, EpochId::new(10))
                    .unwrap()
            }),
            1
        );
        assert!(store.contains(&candidate));
        assert_eq!(store.commit_epoch(), EpochId::new(10));

        let rollback_tid = TransactionId::new(52);
        with_authority(&owner, || {
            store.remove_in_transaction(rollback_tid, candidate.clone());
        });
        assert_eq!(store.rollback_dataset(rollback_tid), 0);
        assert!(store.has_pending_ops(rollback_tid));
        with_authority(&wrong, || {
            assert_eq!(store.rollback_dataset(rollback_tid), 0);
        });
        assert!(store.has_pending_ops(rollback_tid));
        assert_eq!(
            with_authority(&owner, || store.rollback_dataset(rollback_tid)),
            1
        );
        assert!(store.contains(&candidate));

        let snapshot_tid = TransactionId::new(53);
        store.register_transaction_snapshot(snapshot_tid, EpochId::new(10));
        assert!(
            !store
                .tx_buffer
                .read()
                .snapshot_epochs
                .contains_key(&snapshot_tid)
        );
        with_authority(&wrong, || {
            store.register_transaction_snapshot(snapshot_tid, EpochId::new(10));
        });
        assert!(
            !store
                .tx_buffer
                .read()
                .snapshot_epochs
                .contains_key(&snapshot_tid)
        );
        with_authority(&owner, || {
            store.register_transaction_snapshot(snapshot_tid, EpochId::new(10));
        });
        assert_eq!(
            store.tx_buffer.read().snapshot_epochs.get(&snapshot_tid),
            Some(&EpochId::new(10))
        );

        store.mark_projection_rebuilt(1, EpochId::new(10));
        assert_eq!(store.projection_rebuilt_at(1), None);
        with_authority(&wrong, || {
            store.mark_projection_rebuilt(1, EpochId::new(10));
        });
        assert_eq!(store.projection_rebuilt_at(1), None);
        with_authority(&owner, || {
            store.mark_projection_rebuilt(1, EpochId::new(10));
        });
        assert_eq!(store.projection_rebuilt_at(1), Some(EpochId::new(10)));

        // Exact derived cache rebuilds remain usable because they cannot inject
        // caller-supplied logical or transaction state.
        assert_eq!(store.collect_statistics().total_triples, 2);
        assert!(!store.get_or_build_dictionary().is_empty());
        #[cfg(feature = "ring-index")]
        {
            store.rebuild_ring();
            assert!(store.ring().is_some());
        }
    }

    #[test]
    fn sealed_store_authority_is_owner_specific_and_inherited_by_children() {
        use crate::graph::write_permit::{WriteAuthority, with_authority};

        let store_a = RdfStore::new();
        let store_b = RdfStore::new();
        let authority_a = WriteAuthority::new();
        let authority_b = WriteAuthority::new();
        assert!(store_a.seal_unframed_writes(&authority_a));
        assert!(store_b.seal_unframed_writes(&authority_b));

        let triple_a = sample_triples()[0].clone();
        let triple_b = sample_triples()[1].clone();
        with_authority(&authority_a, || {
            assert!(store_a.insert(triple_a.clone()));
            assert!(!store_b.insert(triple_b.clone()));
            assert!(store_a.create_graph("http://example.org/owned-child"));
        });
        with_authority(&authority_b, || {
            assert!(store_b.insert(triple_b.clone()));
            assert!(!store_a.insert(triple_b.clone()));
        });

        let child = store_a.graph("http://example.org/owned-child").unwrap();
        assert!(!child.insert(triple_b.clone()));
        with_authority(&authority_b, || {
            assert!(!child.insert(triple_b.clone()));
        });
        with_authority(&authority_a, || {
            assert!(child.insert(triple_b));
        });
    }

    #[test]
    fn test_insert_and_contains() {
        let store = RdfStore::new();
        let triples = sample_triples();

        for triple in &triples {
            assert!(store.insert(triple.clone()));
        }

        assert_eq!(store.len(), 4);

        for triple in &triples {
            assert!(store.contains(triple));
        }

        // Inserting duplicate should return false
        assert!(!store.insert(triples[0].clone()));
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn test_remove() {
        let store = RdfStore::new();
        let triples = sample_triples();

        for triple in &triples {
            store.insert(triple.clone());
        }

        assert!(store.remove(&triples[0]));
        assert_eq!(store.len(), 3);
        assert!(!store.contains(&triples[0]));

        // Removing non-existent should return false
        assert!(!store.remove(&triples[0]));
    }

    #[test]
    fn test_query_by_subject() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let alix = Term::iri("http://example.org/alix");
        let alice_triples = store.triples_with_subject(&alix);

        assert_eq!(alice_triples.len(), 3);
        for triple in &alice_triples {
            assert_eq!(triple.subject(), &alix);
        }
    }

    #[test]
    fn test_query_by_predicate() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let name_pred = Term::iri("http://xmlns.com/foaf/0.1/name");
        let name_triples = store.triples_with_predicate(&name_pred);

        assert_eq!(name_triples.len(), 2);
        for triple in &name_triples {
            assert_eq!(triple.predicate(), &name_pred);
        }
    }

    #[test]
    fn test_query_by_object() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let gus = Term::iri("http://example.org/gus");
        let bob_triples = store.triples_with_object(&gus);

        assert_eq!(bob_triples.len(), 1);
        assert_eq!(bob_triples[0].object(), &gus);
    }

    #[test]
    fn test_pattern_matching() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // Find all triples with subject alix and predicate knows
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/knows")),
            object: None,
        };

        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::iri("http://example.org/gus"));
    }

    #[test]
    fn test_stats() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let stats = store.stats();
        assert_eq!(stats.triple_count, 4);
        assert_eq!(stats.subject_count, 2); // alix, gus
        assert_eq!(stats.predicate_count, 3); // name, age, knows
    }

    #[test]
    fn test_clear() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        assert!(!store.is_empty());
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_find_with_pending_filters_deletes() {
        let store = RdfStore::new();
        let triples = sample_triples();

        // Insert all triples into committed storage
        for triple in &triples {
            store.insert(triple.clone());
        }

        // Create a transaction and add a pending delete
        let transaction_id = TransactionId::new(1);
        store.remove_in_transaction(transaction_id, triples[0].clone()); // Delete Alix's name triple

        // Query with transaction context - should NOT see the deleted triple
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: None,
            object: None,
        };

        let results = store.find_with_pending(&pattern, Some(transaction_id));
        assert_eq!(results.len(), 2); // Should be 2, not 3 (one deleted)

        // Verify the deleted triple is not in results
        let deleted = &triples[0];
        for result in &results {
            assert_ne!(result.as_ref(), deleted);
        }

        // Query without transaction context - should still see all 3
        let results_no_tx = store.find_with_pending(&pattern, None);
        assert_eq!(results_no_tx.len(), 3);

        // Verify pending inserts are still included
        let new_triple = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/email"),
            Term::literal("alix@example.org"),
        );
        store.insert_in_transaction(transaction_id, new_triple.clone());

        let results_with_insert = store.find_with_pending(&pattern, Some(transaction_id));
        assert_eq!(results_with_insert.len(), 3); // 2 committed - 1 deleted + 1 inserted

        // Verify the new triple is in results
        let found_new = results_with_insert
            .iter()
            .any(|t| t.as_ref() == &new_triple);
        assert!(found_new, "Pending insert should be visible");
    }

    #[test]
    fn named_graph_lifecycle_is_detached_and_savepoint_restorable() {
        let store = RdfStore::new();
        let tx = TransactionId::new(41);
        let existing_name = "http://example.org/existing";
        let created_name = "http://example.org/created";
        let triple = sample_triples()[0].clone();

        assert!(store.create_graph(existing_name));
        let existing = store.graph(existing_name).unwrap();
        assert!(existing.insert(triple.clone()));
        let savepoint = store.transaction_savepoint(tx);

        assert!(store.create_graph_in_tx(created_name, Some(tx)));
        let detached = store
            .graph_in_transaction(created_name, Some(tx))
            .expect("owner sees detached create");
        detached.insert_in_transaction(tx, triple.clone());
        assert!(
            store.graph(created_name).is_none(),
            "uncommitted create must not enter the shared registry"
        );
        assert!(store.drop_graph_in_tx(existing_name, Some(tx)));
        assert!(
            store
                .graph_in_transaction(existing_name, Some(tx))
                .is_none()
        );
        assert!(store.graph(existing_name).is_some(), "drop stays staged");

        store.restore_transaction_savepoint(tx, &savepoint);
        assert!(store.graph_in_transaction(created_name, Some(tx)).is_none());
        assert!(
            store
                .graph_in_transaction(existing_name, Some(tx))
                .is_some()
        );
        assert!(store.validate_transaction_lifecycle(tx).is_ok());
        assert_eq!(store.commit_dataset(tx), 0);
        assert!(store.graph(created_name).is_none());
        assert_eq!(store.graph(existing_name).unwrap().len(), 1);
    }

    #[test]
    fn mutation_target_pins_first_exact_incarnation_across_replacement() {
        let store = RdfStore::new();
        let tx = TransactionId::new(42);
        let name = "http://example.org/mutation-target";
        let before_replacement = labelled_triple("before-replacement");
        let after_replacement = labelled_triple("after-replacement");

        assert!(store.create_graph(name));
        let original = store.graph(name).unwrap();
        let first = store
            .graph_for_mutation_in_tx(name, tx)
            .expect("the existing graph is a mutation target");
        assert!(Arc::ptr_eq(&first, &original));
        first.insert_in_transaction(tx, before_replacement);

        assert!(store.drop_graph(name));
        assert!(store.create_graph(name));
        let replacement = store.graph(name).unwrap();
        assert!(!Arc::ptr_eq(&replacement, &original));

        let repeated_insert_target = store
            .graph_or_create_in_tx(name, Some(tx))
            .expect("insertion retains the transaction's exact first write target");
        assert!(Arc::ptr_eq(&repeated_insert_target, &original));
        let repeated = store
            .graph_for_mutation_in_tx(name, tx)
            .expect("the transaction retains its exact first write target");
        assert!(Arc::ptr_eq(&repeated, &original));
        repeated.insert_in_transaction(tx, after_replacement);

        assert!(store.drop_graph_in_tx(name, Some(tx)));
        let buffer = store.tx_buffer.read();
        let dropped = &buffer.dropped_graphs.get(&tx).unwrap()[name];
        assert!(
            Arc::ptr_eq(&dropped.store, &original),
            "a later DROP must retain the transaction's exact first write target"
        );
        drop(buffer);

        let error = store.validate_transaction_lifecycle(tx).unwrap_err();
        assert!(error.contains("replaced concurrently"), "{error}");
        assert_eq!(store.rollback_dataset(tx), 2);
        assert!(!original.has_pending_ops(tx));
        assert!(replacement.is_empty());
    }

    #[test]
    fn savepoint_retains_deduplicated_detached_touched_and_dropped_partition() {
        let store = RdfStore::new();
        let tx = TransactionId::new(43);
        let name = "http://example.org/detached-before-savepoint";
        let retained = labelled_triple("retained-before-savepoint");
        let discarded = labelled_triple("discarded-after-savepoint");

        assert!(store.create_graph(name));
        let original = store
            .graph_for_mutation_in_tx(name, tx)
            .expect("the graph exists before its staged drop");
        original.insert_in_transaction(tx, retained.clone());
        assert!(store.drop_graph_in_tx(name, Some(tx)));

        assert!(store.drop_graph(name));
        assert!(store.create_graph(name));
        let replacement = store.graph(name).unwrap();
        assert!(!Arc::ptr_eq(&replacement, &original));

        let savepoint = store.transaction_savepoint(tx);
        assert_eq!(
            savepoint
                .named
                .iter()
                .filter(|saved| Arc::ptr_eq(&saved.store, &original))
                .count(),
            1,
            "the touched and dropped pins must retain one exact partition"
        );

        original.insert_in_transaction(tx, discarded.clone());
        store.restore_transaction_savepoint(tx, &savepoint);
        let restored = original.find_with_pending(&TriplePattern::any(), Some(tx));
        assert!(restored.iter().any(|triple| triple.as_ref() == &retained));
        assert!(
            restored.iter().all(|triple| triple.as_ref() != &discarded),
            "restore must rewind the detached partition to its captured buffer"
        );

        let error = store.validate_transaction_lifecycle(tx).unwrap_err();
        assert!(error.contains("replaced concurrently"), "{error}");
        assert_eq!(store.rollback_dataset(tx), 1);
        assert!(!original.has_pending_ops(tx));
    }

    #[test]
    fn savepoint_restore_cleans_partition_touched_and_detached_after_capture() {
        let store = RdfStore::new();
        let tx = TransactionId::new(44);
        let other_tx = TransactionId::new(45);
        let name = "http://example.org/detached-after-savepoint";
        let discarded = labelled_triple("discarded-detached-write");
        let other_write = labelled_triple("other-transaction-write");
        let savepoint = store.transaction_savepoint(tx);

        assert!(store.create_graph(name));
        let detached = store
            .graph_for_mutation_in_tx(name, tx)
            .expect("the graph created after capture is writable");
        detached.insert_in_transaction(tx, discarded);
        detached.insert_in_transaction(other_tx, other_write);
        assert!(store.drop_graph(name));
        assert!(detached.has_pending_ops(tx));

        store.restore_transaction_savepoint(tx, &savepoint);
        assert!(
            !detached.has_pending_ops(tx),
            "restore must clean the post-capture write from its now-detached exact partition"
        );
        assert!(
            detached.has_pending_ops(other_tx),
            "restore must not disturb another transaction on the same partition"
        );
        let buffer = store.tx_buffer.read();
        assert!(!buffer.touched_graphs.contains_key(&tx));
        drop(buffer);
        assert_eq!(detached.rollback_transaction(other_tx), 1);
    }

    #[test]
    fn rollback_dataset_cleans_detached_touched_and_dropped_partition() {
        let store = RdfStore::new();
        let tx = TransactionId::new(46);
        let name = "http://example.org/detached-rollback";

        assert!(store.create_graph(name));
        let detached = store
            .graph_for_mutation_in_tx(name, tx)
            .expect("the graph exists before rollback");
        detached.insert_in_transaction(tx, labelled_triple("rolled-back"));
        assert!(store.drop_graph_in_tx(name, Some(tx)));
        assert!(store.drop_graph(name));

        assert_eq!(store.rollback_dataset(tx), 1);
        assert!(!detached.has_pending_ops(tx));
        let buffer = store.tx_buffer.read();
        assert!(!buffer.created_graphs.contains_key(&tx));
        assert!(!buffer.dropped_graphs.contains_key(&tx));
        assert!(!buffer.touched_graphs.contains_key(&tx));
    }

    #[test]
    fn concurrent_named_graph_create_and_drop_use_revision_cas() {
        let store = RdfStore::new();
        let create_a = TransactionId::new(51);
        let create_b = TransactionId::new(52);
        let name = "http://example.org/raced";

        assert!(store.create_graph_in_tx(name, Some(create_a)));
        assert!(store.create_graph_in_tx(name, Some(create_b)));
        store.validate_transaction_lifecycle(create_a).unwrap();
        assert_eq!(store.commit_dataset(create_a), 0);
        let error = store.validate_transaction_lifecycle(create_b).unwrap_err();
        assert!(error.contains("created concurrently"), "{error}");
        store.rollback_dataset(create_b);

        let drop_tx = TransactionId::new(53);
        let writer_tx = TransactionId::new(54);
        assert!(store.drop_graph_in_tx(name, Some(drop_tx)));
        let writer_graph = store.graph_or_create_in_tx(name, Some(writer_tx)).unwrap();
        writer_graph.insert_in_transaction(writer_tx, sample_triples()[0].clone());
        store.validate_transaction_lifecycle(writer_tx).unwrap();
        assert_eq!(store.commit_dataset(writer_tx), 1);

        let error = store.validate_transaction_lifecycle(drop_tx).unwrap_err();
        assert!(error.contains("changed concurrently"), "{error}");
        store.rollback_dataset(drop_tx);
        assert_eq!(store.graph(name).unwrap().len(), 1);
    }

    #[test]
    fn snapshot_named_graph_reads_pin_identity_absence_and_catalog() {
        let store = RdfStore::new();
        let tx = TransactionId::new(55);
        let existing = "http://example.org/existing";
        let missing = "http://example.org/missing";
        let late = "http://example.org/late";

        assert!(store.create_graph(existing));
        let original = store.graph(existing).unwrap();
        store.register_transaction_snapshot(tx, EpochId::new(0));

        let first = store.graph_in_transaction(existing, Some(tx)).unwrap();
        assert!(Arc::ptr_eq(&first, &original));
        assert!(store.graph_in_transaction(missing, Some(tx)).is_none());

        assert!(store.drop_graph(existing));
        assert!(store.create_graph(existing));
        let replacement = store.graph(existing).unwrap();
        assert!(!Arc::ptr_eq(&replacement, &original));
        assert!(store.create_graph(missing));

        let repeated = store.graph_in_transaction(existing, Some(tx)).unwrap();
        assert!(
            Arc::ptr_eq(&repeated, &original),
            "a replacement must not change an existing read pin"
        );
        assert!(
            store.graph_in_transaction(missing, Some(tx)).is_none(),
            "a first observed absence must remain absent"
        );

        let names = store.graph_names_in_transaction(Some(tx));
        assert!(names.contains(&existing.to_string()));
        assert!(!names.contains(&missing.to_string()));
        assert!(store.create_graph(late));
        assert!(
            !store
                .graph_names_in_transaction(Some(tx))
                .contains(&late.to_string()),
            "catalog enumeration must not admit a later graph phantom"
        );

        // A later write must use the same old incarnation and retain the
        // existing lifecycle CAS conflict behavior.
        let write_target = store.graph_or_create_in_tx(existing, Some(tx)).unwrap();
        assert!(Arc::ptr_eq(&write_target, &original));
        let error = store.validate_transaction_lifecycle(tx).unwrap_err();
        assert!(error.contains("replaced concurrently"), "{error}");
        store.rollback_dataset(tx);
        {
            let buffer = store.tx_buffer.read();
            assert!(!buffer.read_graphs.contains_key(&tx));
            assert!(!buffer.snapshotted_graph_catalogs.contains(&tx));
        }

        // Read Committed transactions have no registered snapshot and keep
        // resolving the live registry on every statement.
        let read_committed = TransactionId::new(56);
        assert!(Arc::ptr_eq(
            &store
                .graph_in_transaction(existing, Some(read_committed))
                .unwrap(),
            &replacement
        ));
    }

    #[test]
    fn transaction_snapshot_reads_start_epoch_and_own_writes() {
        let store = RdfStore::new();
        let tx = TransactionId::new(7);
        let pattern = TriplePattern::any();
        let before = Triple::new(
            Term::iri("http://example.org/before"),
            Term::iri("http://example.org/p"),
            Term::literal("before"),
        );
        let after = Triple::new(
            Term::iri("http://example.org/after"),
            Term::iri("http://example.org/p"),
            Term::literal("after"),
        );
        let own = Triple::new(
            Term::iri("http://example.org/own"),
            Term::iri("http://example.org/p"),
            Term::literal("own"),
        );

        store.try_set_commit_epoch(EpochId::new(1)).unwrap();
        assert!(store.insert(before.clone()));
        store.register_transaction_snapshot(tx, EpochId::new(1));

        store.try_set_commit_epoch(EpochId::new(2)).unwrap();
        assert!(store.remove(&before));
        assert!(store.insert(after.clone()));

        let snapshot = store.find_with_pending(&pattern, Some(tx));
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].as_ref(), &before);

        store.remove_in_transaction(tx, before);
        store.insert_in_transaction(tx, own.clone());
        let with_own_writes = store.find_with_pending(&pattern, Some(tx));
        assert_eq!(with_own_writes.len(), 1);
        assert_eq!(with_own_writes[0].as_ref(), &own);

        let read_committed = store.find_with_pending(&pattern, None);
        assert_eq!(read_committed.len(), 1);
        assert_eq!(read_committed[0].as_ref(), &after);
    }

    #[test]
    fn named_graphs_inherit_transaction_snapshot() {
        let store = RdfStore::new();
        let tx = TransactionId::new(8);
        let graph_name = "http://example.org/g";
        let graph = store.graph_or_create(graph_name).unwrap();
        let before = Triple::new(
            Term::iri("http://example.org/before"),
            Term::iri("http://example.org/p"),
            Term::literal("before"),
        );
        let after = Triple::new(
            Term::iri("http://example.org/after"),
            Term::iri("http://example.org/p"),
            Term::literal("after"),
        );

        store.try_set_commit_epoch(EpochId::new(1)).unwrap();
        assert!(graph.insert(before.clone()));
        store.register_transaction_snapshot(tx, EpochId::new(1));

        store.try_set_commit_epoch(EpochId::new(2)).unwrap();
        assert!(graph.insert(after));

        let rows =
            store.find_in_graphs_with_pending(&TriplePattern::any(), Some(&[graph_name]), Some(tx));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.as_ref(), &before);

        // A graph created after registration also inherits the same cut.
        let late_name = "http://example.org/late";
        let late = store.graph_or_create(late_name).unwrap();
        late.try_set_commit_epoch(EpochId::new(2)).unwrap();
        assert!(late.insert(Triple::new(
            Term::iri("http://example.org/late"),
            Term::iri("http://example.org/p"),
            Term::literal("late"),
        )));
        assert!(
            store
                .find_in_graphs_with_pending(&TriplePattern::any(), Some(&[late_name]), Some(tx),)
                .is_empty()
        );
    }

    #[test]
    fn test_named_graph_crud() {
        let store = RdfStore::new();

        // Create named graph
        assert!(store.create_graph("http://example.org/g1"));
        assert!(!store.create_graph("http://example.org/g1")); // already exists
        assert_eq!(store.graph_count(), 1);

        // Insert into named graph
        let g1 = store.graph("http://example.org/g1").unwrap();
        g1.insert(Triple::new(
            Term::iri("http://example.org/s1"),
            Term::iri("http://example.org/p1"),
            Term::literal("o1"),
        ));
        assert_eq!(g1.len(), 1);

        // Default graph is still empty
        assert_eq!(store.len(), 0);

        // Query named graph
        let results = g1.find(&TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        });
        assert_eq!(results.len(), 1);

        // Drop graph
        assert!(store.drop_graph("http://example.org/g1"));
        assert!(!store.drop_graph("http://example.org/g1"));
        assert_eq!(store.graph_count(), 0);
    }

    #[test]
    fn test_named_graph_isolation() {
        let store = RdfStore::new();

        // Insert into default graph
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("default"),
        ));

        // Insert into named graph
        let g1 = store.graph_or_create("http://example.org/g1").unwrap();
        g1.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("named"),
        ));

        // Each graph sees only its own triples
        assert_eq!(store.len(), 1);
        assert_eq!(g1.len(), 1);
        assert_eq!(store.triples()[0].object(), &Term::literal("default"));
        assert_eq!(g1.triples()[0].object(), &Term::literal("named"));
    }

    #[test]
    fn test_find_in_graphs() {
        let store = RdfStore::new();
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };

        store.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("default"),
        ));

        let g1 = store.graph_or_create("http://example.org/g1").unwrap();
        g1.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("g1"),
        ));

        // Default only
        let results = store.find_in_graphs(&pattern, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_none());

        // All named graphs (excludes default)
        let results = store.find_in_graphs(&pattern, Some(&[]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_deref(), Some("http://example.org/g1"));

        // Specific named graph
        let results = store.find_in_graphs(&pattern, Some(&["http://example.org/g1"]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_deref(), Some("http://example.org/g1"));
    }

    #[test]
    fn test_copy_move_add_graph() {
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        );

        // Insert into default graph
        store.insert(triple.clone());

        // Copy default -> named
        store.copy_graph(None, Some("http://example.org/copy"));
        assert_eq!(store.len(), 1); // default still has it
        let copy = store.graph("http://example.org/copy").unwrap();
        assert_eq!(copy.len(), 1);

        // Add named -> another named (union)
        let g2 = store.graph_or_create("http://example.org/g2").unwrap();
        g2.insert(Triple::new(
            Term::iri("http://example.org/s2"),
            Term::iri("http://example.org/p2"),
            Term::literal("extra"),
        ));
        store.add_graph(
            Some("http://example.org/copy"),
            Some("http://example.org/g2"),
        );
        assert_eq!(g2.len(), 2); // original + added

        // Move named -> named
        store.move_graph(Some("http://example.org/g2"), Some("http://example.org/g3"));
        assert!(store.graph("http://example.org/g2").is_none());
        let g3 = store.graph("http://example.org/g3").unwrap();
        assert_eq!(g3.len(), 2);
    }

    #[test]
    fn test_transaction_commit_and_rollback() {
        let store = RdfStore::new();
        let triples = sample_triples();

        // Insert initial triples
        for triple in &triples {
            store.insert(triple.clone());
        }
        assert_eq!(store.len(), 4);

        // Test rollback
        let tx1 = TransactionId::new(1);
        store.remove_in_transaction(tx1, triples[0].clone());
        assert!(store.has_pending_ops(tx1));

        let discarded = store.rollback_transaction(tx1);
        assert_eq!(discarded, 1);
        assert!(!store.has_pending_ops(tx1));
        assert_eq!(store.len(), 4); // No change

        // Test commit
        let tx2 = TransactionId::new(2);
        store.remove_in_transaction(tx2, triples[0].clone());

        let applied = store.commit_transaction(tx2);
        assert_eq!(applied, 1);
        assert_eq!(store.len(), 3); // Triple removed
        assert!(!store.contains(&triples[0]));
    }

    #[test]
    fn test_batch_insert() {
        let store = RdfStore::new();
        let triples = sample_triples();

        let inserted = store.batch_insert(triples.clone());
        assert_eq!(inserted, 4);
        assert_eq!(store.len(), 4);

        // All triples should be queryable
        for triple in &triples {
            assert!(store.contains(triple));
        }

        // Indexes should be populated correctly
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);
    }

    #[test]
    fn test_batch_insert_with_duplicates() {
        let store = RdfStore::new();

        // Insert one triple first
        let triples = sample_triples();
        store.insert(triples[0].clone());
        assert_eq!(store.len(), 1);

        // Batch insert all 4: only 3 should be new
        let inserted = store.batch_insert(triples.clone());
        assert_eq!(inserted, 3);
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn test_batch_insert_empty() {
        let store = RdfStore::new();
        let inserted = store.batch_insert(Vec::<Triple>::new());
        assert_eq!(inserted, 0);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_composite_index_sp_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // S+P bound: should use SP composite index (no filtering)
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::literal("Alix"));
    }

    #[test]
    fn test_composite_index_po_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // P+O bound: should use PO composite index
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Alix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject(), &Term::iri("http://example.org/alix"));
    }

    #[test]
    fn test_composite_index_os_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // S+O bound: should use OS composite index
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: None,
            object: Some(Term::iri("http://example.org/gus")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].predicate(),
            &Term::iri("http://xmlns.com/foaf/0.1/knows")
        );
    }

    #[test]
    fn test_composite_index_spo_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // S+P+O fully bound: existence check via SP composite
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Alix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);

        // Non-existent triple
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("NotAlix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_composite_index_removal() {
        let store = RdfStore::new();
        let triples = sample_triples();
        for triple in &triples {
            store.insert(triple.clone());
        }

        // Remove alix's name triple
        store.remove(&triples[0]);

        // SP lookup should no longer find it
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 0);

        // PO lookup should only find gus's name
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Alix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_composite_index_batch_insert() {
        let store = RdfStore::new();
        let triples = sample_triples();
        store.batch_insert(triples);

        // Verify composite indexes are populated by batch_insert
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/knows")),
            object: None,
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::iri("http://example.org/gus"));

        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Gus")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bulk_load() {
        let store = RdfStore::new();

        // Insert some existing data that should be replaced
        store.insert(Triple::new(
            Term::iri("http://example.org/old"),
            Term::iri("http://example.org/p"),
            Term::literal("old"),
        ));

        let result = store.bulk_load(sample_triples());
        assert_eq!(result.triple_count, 4);
        assert_eq!(store.len(), 4);

        // Old data should be gone
        assert!(!store.contains(&Triple::new(
            Term::iri("http://example.org/old"),
            Term::iri("http://example.org/p"),
            Term::literal("old"),
        )));

        // All indexes should work (single-term)
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);

        // Composite indexes should work
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(store.find(&pattern).len(), 1);

        // Statistics should be computed
        assert_eq!(result.statistics.total_triples, 4);
        assert_eq!(result.statistics.subject_count, 2);
        assert_eq!(result.statistics.predicate_count, 3);
    }

    #[test]
    fn test_bulk_load_empty() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("v"),
        ));

        let result = store.bulk_load(Vec::<Triple>::new());
        assert_eq!(result.triple_count, 0);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_parse_ntriples_line() {
        // Simple IRI triple
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" ."#,
        );
        assert!(triple.is_some());
        let triple = triple.unwrap();
        assert_eq!(triple.subject(), &Term::iri("http://example.org/alix"));
        assert_eq!(
            triple.predicate(),
            &Term::iri("http://xmlns.com/foaf/0.1/name")
        );
        assert_eq!(triple.object(), &Term::literal("Alix"));

        // Typed literal
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> ."#,
        );
        assert!(triple.is_some());
        let triple = triple.unwrap();
        assert_eq!(
            triple.object(),
            &Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer")
        );

        // Language-tagged literal
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix"@en ."#,
        );
        assert!(triple.is_some());
        assert_eq!(triple.unwrap().object(), &Term::lang_literal("Alix", "en"));

        // Blank node subject
        let triple = parse_ntriples_line(r#"_:b0 <http://xmlns.com/foaf/0.1/name> "Gus" ."#);
        assert!(triple.is_some());
        assert_eq!(triple.unwrap().subject(), &Term::blank("b0"));

        // IRI object
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> ."#,
        );
        assert!(triple.is_some());
        assert_eq!(
            triple.unwrap().object(),
            &Term::iri("http://example.org/gus")
        );

        // Invalid line (no dot)
        assert!(
            parse_ntriples_line(r#"<http://example.org/s> <http://example.org/p> "v""#,).is_none()
        );
    }

    #[test]
    fn test_load_ntriples() {
        let ntriples = "\
<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> \"Alix\" .
# This is a comment
<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .

<http://example.org/gus> <http://xmlns.com/foaf/0.1/name> \"Gus\" .
";
        let store = RdfStore::new();
        let result = store.load_ntriples(ntriples.as_bytes()).unwrap();
        assert_eq!(result.triple_count, 3);
        assert_eq!(store.len(), 3);
        assert_eq!(result.statistics.total_triples, 3);

        // Verify composite index works after load
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Gus")),
        };
        assert_eq!(store.find(&pattern).len(), 1);
    }

    #[test]
    fn test_load_turtle_roundtrip() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alix a foaf:Person ;
                foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;

        let store = RdfStore::new();
        let result = store.load_turtle(turtle).unwrap();
        assert_eq!(result.triple_count, 4);
        assert_eq!(store.len(), 4);

        // Verify subject/predicate indexes are populated.
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);
    }

    #[test]
    fn test_to_turtle_roundtrip() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alix foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;

        let store = RdfStore::new();
        store.load_turtle(turtle).unwrap();
        assert_eq!(store.len(), 3);

        // Serialize to Turtle and re-parse.
        let output = store.to_turtle().unwrap();
        assert!(!output.is_empty());

        let store2 = RdfStore::new();
        let result2 = store2.load_turtle(&output).unwrap();
        assert_eq!(result2.triple_count, 3);

        // Verify structural equivalence: same subject/predicate/object triples exist.
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store2.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::literal("Alix"));

        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/gus")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store2.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::literal("Gus"));
    }

    #[test]
    fn test_load_ntriples_parse_error() {
        let bad_ntriples = "\
<http://example.org/s> <http://example.org/p> \"ok\" .
this is not valid ntriples
<http://example.org/s2> <http://example.org/p2> \"ok2\" .
";
        let store = RdfStore::new();
        let result = store.load_ntriples(bad_ntriples.as_bytes());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            NTriplesError::Parse { line, .. } => assert_eq!(line, 2),
            _ => panic!("expected Parse error"),
        }
    }

    // =========================================================================
    // Streaming load tests (TripleSink-based)
    // =========================================================================

    #[test]
    fn test_load_turtle_streaming_inserts_incrementally() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alix a foaf:Person ;
                foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 2).unwrap();
        assert_eq!(count, 4);
        assert_eq!(store.len(), 4);

        // Verify indexes work
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);
    }

    #[test]
    fn test_load_turtle_streaming_does_not_replace_existing() {
        let store = RdfStore::new();
        // Pre-load one triple
        store.insert(Triple::new(
            Term::iri("http://example.org/existing"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        ));
        assert_eq!(store.len(), 1);

        let turtle = r#"
            <http://example.org/new> <http://example.org/p> "added" .
        "#;
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        assert_eq!(count, 1);
        // Both the existing and new triple should be present
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_load_turtle_streaming_deduplicates() {
        let turtle = r#"
            <http://example.org/s> <http://example.org/p> "o" .
            <http://example.org/s> <http://example.org/p> "o" .
            <http://example.org/s> <http://example.org/p> "o" .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        assert_eq!(count, 1, "duplicates should be filtered by batch_insert");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_load_ntriples_streaming() {
        let ntriples = "\
<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> \"Alix\" .
<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
<http://example.org/gus> <http://xmlns.com/foaf/0.1/name> \"Gus\" .
";
        let store = RdfStore::new();
        let count = store
            .load_ntriples_streaming(ntriples.as_bytes(), 2)
            .unwrap();
        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_load_ntriples_streaming_does_not_replace_existing() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://example.org/existing"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        ));

        let ntriples = "<http://example.org/new> <http://example.org/p> \"added\" .\n";
        let count = store
            .load_ntriples_streaming(ntriples.as_bytes(), 100)
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_load_turtle_reader() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alix ex:name "Alix" .
            ex:gus ex:name "Gus" .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_reader(turtle.as_bytes(), 100).unwrap();
        assert_eq!(count, 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_parse_into_with_count_sink() {
        use crate::graph::rdf::sink::CountSink;
        use crate::graph::rdf::turtle::TurtleParser;

        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:a ex:p "1" .
            ex:b ex:p "2" .
            ex:c ex:p "3" .
        "#;

        let mut sink = CountSink::new();
        let mut parser = TurtleParser::new();
        parser.parse_into(turtle, &mut sink).unwrap();
        assert_eq!(sink.count(), 3);
    }

    #[test]
    fn test_streaming_turtle_with_collections() {
        // Collections emit rdf:first/rdf:rest triples through the sink
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:list ex:items ( "a" "b" "c" ) .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        // 1 (ex:list ex:items _:head) + 3*(first+rest) = 7
        assert_eq!(count, 7);
        assert_eq!(store.len(), 7);
    }

    #[test]
    fn test_streaming_turtle_with_blank_node_property_list() {
        // Blank node property lists emit triples through the sink
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alix ex:address [ ex:city "Amsterdam" ; ex:country "NL" ] .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        // 1 (alix address _:b) + 2 (city, country) = 3
        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn test_ring_built_during_bulk_load() {
        let store = RdfStore::new();
        assert!(
            store.ring().is_none(),
            "ring should not exist before bulk load"
        );

        store.bulk_load(vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/knows"),
                Term::iri("http://ex.org/gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/knows"),
                Term::iri("http://ex.org/alix"),
            ),
        ]);

        let ring = store.ring().expect("ring should exist after bulk load");
        assert_eq!(ring.len(), 2);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn test_ring_stale_after_insert() {
        let store = RdfStore::new();
        store.bulk_load(vec![Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::iri("http://ex.org/b"),
        )]);
        assert!(store.ring().is_some());

        // Incremental insert marks ring stale
        store.insert(Triple::new(
            Term::iri("http://ex.org/c"),
            Term::iri("http://ex.org/p"),
            Term::iri("http://ex.org/d"),
        ));
        assert!(
            store.ring().is_none(),
            "ring should be stale after incremental insert"
        );

        // Rebuild restores ring
        store.rebuild_ring();
        let ring = store.ring().expect("ring should exist after rebuild");
        assert_eq!(ring.len(), 2);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn test_ring_find_matches_store_find() {
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://ex.org/gus"),
            ),
        ];

        let store = RdfStore::new();
        store.bulk_load(triples);

        let ring = store.ring().expect("ring should exist");

        // Fully unbound: ring.count should match store.len
        let all_pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        assert_eq!(ring.count(&all_pattern), store.len());

        // Predicate-bound: count names
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(ring.count(&name_pattern), 2);

        // Subject-bound: triples about alix
        let alix_pattern = TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: None,
            object: None,
        };
        assert_eq!(ring.count(&alix_pattern), 2);
    }

    #[test]
    fn test_batch_insert_composite_indexes() {
        let store = RdfStore::new();
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/age"),
                Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
            ),
        ];
        let inserted = store.batch_insert(triples);
        assert_eq!(inserted, 3);
        assert_eq!(store.len(), 3);

        // Verify composite indexes via find
        let sp_result = store.find(&TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: Some(Term::iri("http://ex.org/name")),
            object: None,
        });
        assert_eq!(sp_result.len(), 1);
        let po_result = store.find(&TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://ex.org/name")),
            object: Some(Term::literal("Gus")),
        });
        assert_eq!(po_result.len(), 1);
        let os_result = store.find(&TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: None,
            object: Some(Term::literal("Alix")),
        });
        assert_eq!(os_result.len(), 1);
    }

    #[test]
    fn test_batch_insert_deduplication() {
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/b"),
            Term::literal("c"),
        );
        let inserted = store.batch_insert(vec![triple.clone(), triple.clone(), triple]);
        assert_eq!(inserted, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_composite_index_after_remove() {
        let store = RdfStore::new();
        let t1 = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v1"),
        );
        let t2 = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v2"),
        );
        store.insert(t1.clone());
        store.insert(t2);
        assert_eq!(
            store
                .find(&TriplePattern {
                    subject: Some(Term::iri("http://ex.org/a")),
                    predicate: Some(Term::iri("http://ex.org/p")),
                    object: None
                })
                .len(),
            2
        );
        store.remove(&t1);
        assert_eq!(
            store
                .find(&TriplePattern {
                    subject: Some(Term::iri("http://ex.org/a")),
                    predicate: Some(Term::iri("http://ex.org/p")),
                    object: None
                })
                .len(),
            1
        );
    }

    #[test]
    fn test_named_graph_operations() {
        let store = RdfStore::new();
        assert!(store.create_graph("http://ex.org/g1"));
        assert!(!store.create_graph("http://ex.org/g1")); // already exists
        let g = store.graph("http://ex.org/g1").unwrap();
        g.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/b"),
            Term::literal("c"),
        ));
        assert_eq!(g.len(), 1);
        assert!(store.drop_graph("http://ex.org/g1"));
        assert!(store.graph("http://ex.org/g1").is_none());
    }

    #[test]
    fn test_statistics_cache_invalidation() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        ));
        let stats1 = store.get_or_collect_statistics();
        // Term::iri.to_string() produces "<http://ex.org/p>"
        assert!(stats1.get_predicate("<http://ex.org/p>").is_some());
        // Insert more data: cache should be invalidated
        store.insert(Triple::new(
            Term::iri("http://ex.org/b"),
            Term::iri("http://ex.org/q"),
            Term::literal("w"),
        ));
        let stats2 = store.get_or_collect_statistics();
        assert!(stats2.get_predicate("<http://ex.org/q>").is_some());
    }

    #[test]
    fn test_find_three_bound() {
        let store = RdfStore::new();
        let t = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        );
        store.insert(t.clone());
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("other"),
        ));
        let result = store.find(&TriplePattern {
            subject: Some(Term::iri("http://ex.org/a")),
            predicate: Some(Term::iri("http://ex.org/p")),
            object: Some(Term::literal("v")),
        });
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_ref(), &t);
    }

    #[test]
    fn find_uses_canonical_language_identity_after_index_narrowing() {
        let store = RdfStore::new();
        for language in ["EN", "en"] {
            store.insert(Triple::new(
                Term::iri("urn:s"),
                Term::iri("urn:p"),
                Term::lang_literal("x", language),
            ));
        }
        store.insert(Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::lang_literal("other", "en"),
        ));
        let pattern = TriplePattern {
            subject: Some(Term::iri("urn:s")),
            predicate: Some(Term::iri("urn:p")),
            object: Some(Term::lang_literal("x", "eN")),
        };

        assert_eq!(store.find(&pattern).len(), 1);
    }

    fn language_alias(language: &str) -> Triple {
        Triple::new(
            Term::iri("urn:s"),
            Term::iri("urn:p"),
            Term::lang_literal("x", language),
        )
    }

    #[test]
    fn canonical_language_aliases_form_one_abstract_graph_statement() {
        for index_objects in [false, true] {
            let store = RdfStore::with_config(RdfStoreConfig {
                index_objects,
                ..RdfStoreConfig::default()
            });
            let upper = language_alias("EN");
            let lower = language_alias("en");
            assert!(store.insert(upper.clone()));
            assert!(!store.insert(lower.clone()));
            assert_eq!(store.len(), 1);
            assert!(store.contains(&lower));
            assert_eq!(
                store.triples_with_object(lower.object()).as_slice(),
                &[Arc::new(upper.clone())]
            );
            assert_eq!(
                store
                    .find(&TriplePattern {
                        subject: None,
                        predicate: Some(Term::iri("urn:p")),
                        object: Some(lower.object().clone())
                    })
                    .len(),
                1
            );
            #[cfg(feature = "ring-index")]
            {
                store.rebuild_ring();
                assert_eq!(store.ring().unwrap().len(), 1);
            }
            assert!(store.remove(&lower));
            assert!(!store.contains(&upper));
            assert!(store.find(&TriplePattern::any()).is_empty());
            assert!(store.triples_with_subject(upper.subject()).is_empty());
            assert!(store.triples_with_predicate(upper.predicate()).is_empty());
            assert!(store.triples_with_object(upper.object()).is_empty());
            assert!(store.insert(lower.clone()));
            assert_eq!(store.triples().as_slice(), &[Arc::new(lower)]);
        }
    }

    #[test]
    fn canonical_membership_preserves_term_kinds_and_literal_lexical_identity() {
        let store = RdfStore::new();
        let make = |object| Triple::new(Term::blank("s"), Term::iri("urn:p"), object);
        let plain = make(Term::literal("x\\\"\\n"));
        let typed = make(Term::typed_literal(
            "x\\\"\\n",
            super::super::Literal::XSD_STRING,
        ));
        assert!(store.insert(plain.clone()));
        assert!(!store.insert(typed.clone()));
        assert!(store.contains(&typed));
        let distinct = [
            make(Term::iri("x\\\"\\n")),
            make(Term::blank("x\\\"\\n")),
            make(Term::typed_literal("1", super::super::Literal::XSD_INTEGER)),
            make(Term::typed_literal(
                "01",
                super::super::Literal::XSD_INTEGER,
            )),
            make(Term::lang_literal("x\\\"\\n", "en")),
            make(Term::lang_literal("x\\\"\\n", "fr")),
        ];
        assert_eq!(store.batch_insert(distinct.clone()), distinct.len());
        assert_eq!(store.len(), distinct.len() + 1);
        for triple in distinct {
            assert!(store.contains(&triple));
            assert!(store.remove(&triple));
        }
        assert!(store.remove(&typed));
        assert!(store.is_empty());
    }

    #[test]
    fn canonical_batch_bulk_and_pending_validity_retain_first_representative() {
        let store = RdfStore::new();
        let upper = language_alias("EN");
        let lower = language_alias("en");
        assert_eq!(store.batch_insert([upper.clone(), lower.clone()]), 1);
        assert_eq!(
            store.bulk_load([lower.clone(), upper.clone()]).triple_count,
            1
        );
        assert_eq!(store.triples().as_slice(), &[Arc::new(lower.clone())]);
        store.clear();
        let tx = TransactionId::new(91);
        let first = ValidTimeInterval::from_tai_nanoseconds(1, 2).unwrap();
        let later = ValidTimeInterval::from_tai_nanoseconds(3, 4).unwrap();
        store.insert_in_transaction_with_valid(tx, upper.clone(), Some(first));
        store.insert_in_transaction_with_valid(tx, lower.clone(), Some(later));
        assert_eq!(
            store.visible_with_valid(Some(tx)),
            [(upper.clone(), Some(first))]
        );
        store.remove_in_transaction(tx, lower.clone());
        assert!(
            store
                .find_with_pending(&TriplePattern::any(), Some(tx))
                .is_empty()
        );
        store.insert_in_transaction_with_valid(tx, lower.clone(), Some(later));
        assert_eq!(
            store.visible_with_valid(Some(tx)),
            [(lower.clone(), Some(later))]
        );
        store.try_commit_dataset_at(tx, EpochId::new(1)).unwrap();
        assert_eq!(store.visible_with_valid(None), [(lower, Some(later))]);
    }

    #[test]
    fn canonical_writer_revision_survives_savepoint_and_rejects_competing_alias() {
        let store = RdfStore::new();
        let first = TransactionId::new(92);
        let second = TransactionId::new(93);
        store.register_transaction_snapshot(first, EpochId::new(0));
        store.register_transaction_snapshot(second, EpochId::new(0));
        let saved = store.transaction_savepoint(second);
        store.insert_in_transaction(second, language_alias("en"));
        store.restore_transaction_savepoint(second, &saved);
        store.insert_in_transaction(first, language_alias("EN"));
        store.validate_transaction_lifecycle(first).unwrap();
        store.try_commit_dataset_at(first, EpochId::new(1)).unwrap();
        // A restored read-only snapshot remains valid until it tries to write.
        store.validate_transaction_lifecycle(second).unwrap();
        store.insert_in_transaction(second, language_alias("en"));
        assert!(store.validate_transaction_lifecycle(second).is_err());
        store.rollback_dataset(second);
        store.insert_in_transaction(second, language_alias("en"));
        store.validate_transaction_lifecycle(second).unwrap();
    }
}
