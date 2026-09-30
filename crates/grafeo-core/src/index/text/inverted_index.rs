//! BM25-scored inverted index for full-text search.

use super::tokenizer::{SimpleTokenizer, Tokenizer};
use super::versioned::{
    AggDelta, VersionedDocLen, VersionedPosting, doc_len_visible, posting_visible,
};
use grafeo_common::types::{EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashSet;
#[cfg(feature = "lpg")]
use parking_lot::RwLockWriteGuard;
use parking_lot::{ArcRwLockReadGuard, RawRwLock, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "lpg")]
mod commit;
#[cfg(feature = "lpg")]
pub(crate) use commit::{TextCommitScope, TextCommitWorkspace};

/// Configuration for BM25 scoring.
#[derive(Debug, Clone)]
pub struct BM25Config {
    /// Term frequency saturation parameter (default 1.2).
    ///
    /// Higher values give more weight to term frequency.
    pub k1: f64,
    /// Length normalization parameter (default 0.75).
    ///
    /// 0.0 = no length normalization, 1.0 = full normalization.
    pub b: f64,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// A posting list for a single term.
#[derive(Debug, Clone, Default)]
struct PostingList {
    postings: Vec<VersionedPosting>,
}

/// Stable tokenizer identity carried by an exact text-index image.
///
/// Trait-object tokenizers deliberately have no implicit persistence contract:
/// assigning a name to an opaque implementation would silently change search
/// semantics after reopen. New built-ins must therefore add an explicit,
/// lossless descriptor and codec here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum ExactTokenizerDescriptor {
    Simple { min_token_length: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ExactPosting {
    pub(super) node_id: NodeId,
    pub(super) term_freq: u32,
    pub(super) created_epoch: EpochId,
    pub(super) created_by: Option<TransactionId>,
    pub(super) deleted_epoch: Option<EpochId>,
    pub(super) deleted_by: Option<TransactionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ExactPostingList {
    pub(super) term: String,
    pub(super) postings: Vec<ExactPosting>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ExactDocLength {
    pub(super) len: u32,
    pub(super) created_epoch: EpochId,
    pub(super) created_by: Option<TransactionId>,
    pub(super) deleted_epoch: Option<EpochId>,
    pub(super) deleted_by: Option<TransactionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ExactDocHistory {
    pub(super) node_id: NodeId,
    pub(super) history: Vec<ExactDocLength>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ExactAggDelta {
    pub(super) epoch: EpochId,
    pub(super) tx: Option<TransactionId>,
    pub(super) d_total_len: i64,
    pub(super) d_doc_count: i64,
}

/// Complete committed text-index state persisted by the current Text section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ExactTextIndexImage {
    pub(super) k1: f64,
    pub(super) b: f64,
    pub(super) tokenizer: ExactTokenizerDescriptor,
    pub(super) retained_from: EpochId,
    pub(super) postings: Vec<ExactPostingList>,
    pub(super) doc_lengths: Vec<ExactDocHistory>,
    pub(super) agg_log: Vec<ExactAggDelta>,
}

/// Fully validated replacement state. Construction performs every fallible
/// allocation so installation itself is a set of non-failing moves.
pub(super) struct PreparedTextIndexImage {
    config: BM25Config,
    tokenizer: Arc<dyn Tokenizer>,
    tokenizer_descriptor: ExactTokenizerDescriptor,
    retained_from: EpochId,
    postings: HashMap<String, PostingList>,
    doc_lengths: HashMap<NodeId, Vec<VersionedDocLen>>,
    agg_log: Vec<AggDelta>,
}

/// An in-memory inverted index with Okapi BM25 scoring.
///
/// Supports insert, remove, and ranked search operations. Designed
/// for indexing text properties on graph nodes.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "text-index")]
/// # {
/// use grafeo_core::index::text::{InvertedIndex, BM25Config};
/// use grafeo_common::types::NodeId;
///
/// let mut index = InvertedIndex::new(BM25Config::default());
/// index.insert(NodeId::new(1), "rust graph database");
/// index.insert(NodeId::new(2), "python web framework");
///
/// let results = index.search("graph database", 10);
/// assert_eq!(results[0].0, NodeId::new(1));
/// # }
/// ```
pub struct InvertedIndex {
    /// Term → posting list.
    postings: HashMap<String, PostingList>,
    /// Per-document versioned length history.
    ///
    /// Each entry is a list of [`VersionedDocLen`] records (one per insert/
    /// re-insert); only one is live at any given epoch.
    doc_lengths: HashMap<NodeId, Vec<VersionedDocLen>>,
    /// Epoch-stamped aggregate log for O(n_deltas) `total_length@E` /
    /// `doc_count@E` reconstruction without scanning all docs.
    ///
    /// Entries are appended in epoch order; a reader at epoch `E` prefixes-sums
    /// all deltas with `delta.epoch <= E` (or `delta.tx == viewing_tx` for
    /// pending own-tx deltas).
    agg_log: Vec<AggDelta>,
    /// Tokenizer used for indexing and querying.
    tokenizer: Arc<dyn Tokenizer>,
    /// Lossless codec identity for supported built-in tokenizers. `None` is an
    /// opaque custom trait object and must fail closed at persistence time.
    tokenizer_descriptor: Option<ExactTokenizerDescriptor>,
    /// Earliest epoch for which the retained histories are authoritative.
    retained_from: EpochId,
    /// BM25 configuration.
    config: BM25Config,
    /// Store-scoped authority required once this index belongs to a sealed
    /// WAL-backed graph (`0` keeps standalone indexes mutable).
    mutation_scope: AtomicU64,
    /// Per-index barrier retained by every complete mutation and taken
    /// exclusively while the LPG scope is sealed. Keeping this local avoids a
    /// global-text → store versus store → global-text lock-order cycle.
    mutation_scope_gate: Arc<RwLock<()>>,
    /// Stable LPG-store and exact index-slot identity. Unlike `mutation_scope`,
    /// this binds an index even while its owning store remains unsealed,
    /// closing retained-Arc alias sharing between stores or two keys.
    #[cfg(feature = "lpg")]
    owner_store: AtomicU64,
    #[cfg(feature = "lpg")]
    owner_slot: AtomicU64,
    /// Registry-owned authority target for a retained caller handle.
    ///
    /// LPG registration moves the real index into an outer lock that the
    /// caller never receives and leaves this handle forwarding to it. Replacing
    /// the value inside the caller's retained `RwLock` can therefore detach only
    /// that handle; it cannot replace the store's sealed index wholesale.
    #[cfg(feature = "lpg")]
    registry_target: Option<Arc<RwLock<InvertedIndex>>>,
}

/// Viewing epoch used by the committed-latest search path.
///
/// Large enough that all committed postings (epoch 0 … real epochs) are
/// visible, but small enough that PENDING (u64::MAX) inserts are not.
const COMMITTED_EPOCH: EpochId = EpochId::new(u64::MAX - 1);

/// Serializes the rare transition from a standalone text index to a
/// store-owned mutation scope. LPG retains one token across full-tree
/// compatibility preflight and application, closing alias-driven TOCTOU races.
#[cfg(feature = "lpg")]
static TEXT_SCOPE_TRANSITION_GATE: RwLock<()> = RwLock::new(());

/// Proof that no text index can change ownership scope until this value drops.
#[cfg(feature = "lpg")]
pub(crate) struct TextScopeTransition {
    _guard: RwLockWriteGuard<'static, ()>,
}

#[cfg(feature = "lpg")]
impl TextScopeTransition {
    fn acquire() -> Self {
        Self {
            _guard: TEXT_SCOPE_TRANSITION_GATE.write(),
        }
    }
}

/// Retained proof that text-index scope cannot transition during a mutation.
///
/// The per-index read side is recursive so compound public operations can
/// route through a single `_inner` implementation without ever attempting a
/// shared-to-exclusive upgrade. Ordinary mutation never retains the global
/// DDL transition gate.
pub(super) struct TextMutation {
    _guard: ArcRwLockReadGuard<RawRwLock, ()>,
}

impl InvertedIndex {
    /// Creates a new inverted index with the given BM25 configuration.
    #[must_use]
    pub fn new(config: BM25Config) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            agg_log: Vec::new(),
            retained_from: EpochId::INITIAL,
            tokenizer: Arc::new(SimpleTokenizer::new()),
            tokenizer_descriptor: Some(ExactTokenizerDescriptor::Simple {
                min_token_length: 2,
            }),
            config,
            mutation_scope: AtomicU64::new(0),
            mutation_scope_gate: Arc::new(RwLock::new(())),
            #[cfg(feature = "lpg")]
            owner_store: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_slot: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            registry_target: None,
        }
    }

    /// Creates a new inverted index with a custom tokenizer.
    pub fn with_tokenizer(config: BM25Config, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            agg_log: Vec::new(),
            retained_from: EpochId::INITIAL,
            tokenizer: Arc::from(tokenizer),
            tokenizer_descriptor: None,
            config,
            mutation_scope: AtomicU64::new(0),
            mutation_scope_gate: Arc::new(RwLock::new(())),
            #[cfg(feature = "lpg")]
            owner_store: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_slot: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            registry_target: None,
        }
    }

    /// Creates an index with Grafeo's built-in simple tokenizer and an explicit
    /// minimum token length.
    ///
    /// Unlike [`Self::with_tokenizer`], this constructor carries a stable,
    /// lossless persistence descriptor. Opaque custom tokenizers remain usable
    /// in memory but fail closed at the temporary tokenizer-codec boundary.
    #[must_use]
    pub fn with_simple_tokenizer(config: BM25Config, min_token_length: usize) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            agg_log: Vec::new(),
            retained_from: EpochId::INITIAL,
            tokenizer: Arc::new(SimpleTokenizer::with_min_length(min_token_length)),
            tokenizer_descriptor: Some(ExactTokenizerDescriptor::Simple {
                // Rust's supported pointer widths fit losslessly in `u64`.
                // Keep this constructor non-panicking if that ever changes.
                min_token_length: u64::try_from(min_token_length).unwrap_or(u64::MAX),
            }),
            config,
            mutation_scope: AtomicU64::new(0),
            mutation_scope_gate: Arc::new(RwLock::new(())),
            #[cfg(feature = "lpg")]
            owner_store: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            owner_slot: AtomicU64::new(0),
            #[cfg(feature = "lpg")]
            registry_target: None,
        }
    }

    /// Whether this owned value is concrete rather than a moved forwarding shell.
    /// This read-only test exposes neither the target nor mutation authority.
    #[cfg(feature = "lpg")]
    pub(crate) fn is_concrete_registry_candidate(&self) -> bool {
        self.registry_target.is_none()
    }

    /// Moves a registrable index behind a store-owned outer lock and converts
    /// this value into a forwarding handle.
    ///
    /// The returned target is never exposed through the public LPG API. A
    /// repeated registration attempt through the same retained handle returns
    /// the already-pinned target, which also prevents forwarding chains and
    /// cycles from being constructed.
    #[cfg(feature = "lpg")]
    pub(crate) fn pin_registry_target(&mut self) -> Arc<RwLock<InvertedIndex>> {
        if let Some(target) = &self.registry_target {
            if let Some(target_guard) = target.try_read() {
                debug_assert!(
                    target_guard.registry_target.is_none(),
                    "registry targets must be concrete indexes, not forwarding handles"
                );
            }
            return Arc::clone(target);
        }

        // Allocate both shells before moving any caller state. The target shell
        // becomes the concrete index; the forwarding shell remains in the
        // caller-owned outer lock.
        let target = Arc::new(RwLock::new(Self::new(BM25Config::default())));
        let mut forwarding = Self::new(BM25Config::default());
        forwarding.registry_target = Some(Arc::clone(&target));

        {
            let mut target_guard = target.write();
            std::mem::swap(&mut *target_guard, self);
            *self = forwarding;
            debug_assert!(target_guard.registry_target.is_none());
        }
        target
    }

    /// Returns the private concrete target of a retained forwarding handle.
    ///
    /// Callers must retain the forwarding handle's outer lock before resolving
    /// this pointer. That gate prevents registration or wholesale payload
    /// replacement between target discovery and target-lock acquisition.
    pub(super) fn forwarding_target(&self) -> Option<Arc<RwLock<InvertedIndex>>> {
        #[cfg(feature = "lpg")]
        {
            self.registry_target.as_ref().map(Arc::clone)
        }
        #[cfg(not(feature = "lpg"))]
        {
            None
        }
    }

    /// Forks an exact in-memory read image for a retiring LPG generation.
    ///
    /// This is deliberately not a persistence codec: it shares the immutable
    /// tokenizer trait object by identity, so opaque custom tokenizers remain
    /// exact without inventing a name or re-tokenizing their postings. Callers
    /// must hold the registry/generation transition that excludes concurrent
    /// mutation while the maps and histories are cloned.
    ///
    /// Store/slot ownership and the sealed mutation scope are copied. The fork
    /// receives a fresh local mutation gate and no forwarding target; it is a
    /// concrete image suitable for the retired registry only.
    #[cfg(all(feature = "lpg", any(feature = "compact-store", test)))]
    pub(crate) fn exact_runtime_fork(&self) -> Self {
        if let Some(target) = &self.registry_target {
            return target.read().exact_runtime_fork();
        }
        Self {
            postings: self.postings.clone(),
            doc_lengths: self.doc_lengths.clone(),
            agg_log: self.agg_log.clone(),
            retained_from: self.retained_from,
            tokenizer: Arc::clone(&self.tokenizer),
            tokenizer_descriptor: self.tokenizer_descriptor.clone(),
            config: self.config.clone(),
            mutation_scope: AtomicU64::new(self.mutation_scope.load(Ordering::Acquire)),
            mutation_scope_gate: Arc::new(RwLock::new(())),
            owner_store: AtomicU64::new(self.owner_store.load(Ordering::Acquire)),
            owner_slot: AtomicU64::new(self.owner_slot.load(Ordering::Acquire)),
            registry_target: None,
        }
    }

    /// Retains the process-wide text-index scope transition gate.
    #[cfg(feature = "lpg")]
    pub(crate) fn pin_scope_transition() -> TextScopeTransition {
        TextScopeTransition::acquire()
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn binding_is_compatible(
        &self,
        owner: u64,
        slot: u64,
        _transition: &TextScopeTransition,
    ) -> bool {
        if let Some(target) = &self.registry_target {
            return target
                .read()
                .binding_is_compatible(owner, slot, _transition);
        }
        let current_owner = self.owner_store.load(Ordering::Acquire);
        let current_slot = self.owner_slot.load(Ordering::Acquire);
        owner != 0
            && slot != 0
            && ((current_owner == 0 && current_slot == 0)
                || (current_owner == owner && current_slot == slot))
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn is_bound_to(
        &self,
        owner: u64,
        slot: u64,
        _transition: &TextScopeTransition,
    ) -> bool {
        if let Some(target) = &self.registry_target {
            return target.read().is_bound_to(owner, slot, _transition);
        }
        owner != 0
            && slot != 0
            && self.owner_store.load(Ordering::Acquire) == owner
            && self.owner_slot.load(Ordering::Acquire) == slot
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn bind_under_transition(
        &self,
        owner: u64,
        slot: u64,
        _transition: &TextScopeTransition,
    ) -> bool {
        if let Some(target) = &self.registry_target {
            return target
                .read()
                .bind_under_transition(owner, slot, _transition);
        }
        if owner == 0 || slot == 0 {
            return false;
        }
        let current_owner = self.owner_store.load(Ordering::Acquire);
        let current_slot = self.owner_slot.load(Ordering::Acquire);
        if current_owner == owner && current_slot == slot {
            return true;
        }
        if current_owner != 0 || current_slot != 0 {
            return false;
        }
        self.owner_slot.store(slot, Ordering::Release);
        self.owner_store.store(owner, Ordering::Release);
        true
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn scope_is_unsealed(&self, _transition: &TextScopeTransition) -> bool {
        if let Some(target) = &self.registry_target {
            return target.read().scope_is_unsealed(_transition);
        }
        self.mutation_scope.load(Ordering::Acquire) == 0
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn scope_is_compatible(
        &self,
        scope: u64,
        _transition: &TextScopeTransition,
    ) -> bool {
        if let Some(target) = &self.registry_target {
            return target.read().scope_is_compatible(scope, _transition);
        }
        if scope == 0 {
            return false;
        }
        let current = self.mutation_scope.load(Ordering::Acquire);
        current == 0 || current == scope
    }

    #[cfg(feature = "lpg")]
    pub(crate) fn seal_with_scope_under_transition(
        &self,
        scope: u64,
        _transition: &TextScopeTransition,
    ) -> bool {
        if let Some(target) = &self.registry_target {
            return target
                .read()
                .seal_with_scope_under_transition(scope, _transition);
        }
        if scope == 0 {
            return false;
        }
        let _mutation_cut = self.mutation_scope_gate.write();
        match self
            .mutation_scope
            .compare_exchange(0, scope, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(existing) => existing == scope,
        }
    }

    fn pin_mutation(&self) -> Option<TextMutation> {
        let guard = self.mutation_scope_gate.read_arc_recursive();
        let scope = self.mutation_scope.load(Ordering::Acquire);
        let allowed = scope == 0
            || std::num::NonZeroU64::new(scope).is_some_and(crate::graph::write_permit::is_held);
        allowed.then_some(TextMutation { _guard: guard })
    }

    // ── Versioned write path ────────────────────────────────────────────────

    /// Indexes a document stamped with an explicit epoch and optional creator tx.
    ///
    /// If the node is already indexed under a live posting, those postings are
    /// first soft-deleted at `epoch` / `created_by` before the new ones are added.
    pub fn insert_versioned(
        &mut self,
        id: NodeId,
        text: &str,
        epoch: EpochId,
        created_by: Option<TransactionId>,
    ) {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            target.write().insert_versioned(id, text, epoch, created_by);
            return;
        }
        let Some(mutation) = self.pin_mutation() else {
            return;
        };
        self.insert_versioned_inner(id, text, epoch, created_by, &mutation);
    }

    fn insert_versioned_inner(
        &mut self,
        id: NodeId,
        text: &str,
        epoch: EpochId,
        created_by: Option<TransactionId>,
        mutation: &TextMutation,
    ) {
        // Soft-delete any currently-live doc-len entry for this node so
        // re-insertion behaves like update.
        let has_live = self
            .doc_lengths
            .get(&id)
            .is_some_and(|v| v.iter().any(|d| d.deleted_epoch.is_none()));
        if has_live {
            self.remove_versioned_inner(id, epoch, created_by, mutation);
        }

        let tokens = self.tokenizer.tokenize(text);
        // reason: document token count fits u32 for practical text sizes
        #[allow(clippy::cast_possible_truncation)]
        let doc_len = tokens.len() as u32;

        if doc_len == 0 {
            return;
        }

        // Count term frequencies.
        let mut term_freqs: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.as_str()).or_insert(0) += 1;
        }

        // Append versioned postings.
        for (term, freq) in term_freqs {
            self.postings
                .entry(term.to_string())
                .or_default()
                .postings
                .push(VersionedPosting::new(id, freq, epoch, created_by));
        }

        // Append a new versioned doc-length record.
        self.doc_lengths
            .entry(id)
            .or_default()
            .push(VersionedDocLen::new(doc_len, epoch, created_by));

        // Record the aggregate delta.
        self.agg_log.push(AggDelta {
            epoch,
            tx: created_by,
            d_total_len: i64::from(doc_len),
            d_doc_count: 1,
        });
    }

    /// Soft-deletes all live postings for `id` by stamping `deleted_epoch` / `deleted_by`.
    ///
    /// The postings are **retained** — physical cleanup is a future GC step.
    /// Returns `true` if any posting was live (i.e., the document existed).
    pub fn remove_versioned(
        &mut self,
        id: NodeId,
        epoch: EpochId,
        deleted_by: Option<TransactionId>,
    ) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.write().remove_versioned(id, epoch, deleted_by);
        }
        let Some(mutation) = self.pin_mutation() else {
            return false;
        };
        self.remove_versioned_inner(id, epoch, deleted_by, &mutation)
    }

    fn remove_versioned_inner(
        &mut self,
        id: NodeId,
        epoch: EpochId,
        deleted_by: Option<TransactionId>,
        _mutation: &TextMutation,
    ) -> bool {
        // Find the currently-live doc-len entry and soft-delete it.
        let Some(history) = self.doc_lengths.get_mut(&id) else {
            return false;
        };
        let Some(live) = history.iter_mut().find(|d| d.deleted_epoch.is_none()) else {
            return false;
        };
        let doc_len = live.len;
        live.deleted_epoch = Some(epoch);
        live.deleted_by = deleted_by;

        // Record the aggregate delta (negative).
        self.agg_log.push(AggDelta {
            epoch,
            tx: deleted_by,
            d_total_len: -i64::from(doc_len),
            d_doc_count: -1,
        });

        // Stamp deleted_epoch on every live posting for this node.
        for list in self.postings.values_mut() {
            for p in &mut list.postings {
                if p.node_id == id && p.deleted_epoch.is_none() {
                    p.deleted_epoch = Some(epoch);
                    p.deleted_by = deleted_by;
                }
            }
        }

        true
    }

    /// Physically removes every temporal posting and aggregate contribution
    /// belonging to one never-committed identity.
    ///
    /// This is deliberately crate-private and is not ordinary delete: rollback
    /// uses it only after the identity's sole structural lifetime disappears.
    /// It preserves normal temporal removal semantics for committed identities
    /// and performs no allocation.
    #[cfg(feature = "lpg")]
    pub(crate) fn purge_identity(&mut self, id: NodeId) -> bool {
        if let Some(target) = &self.registry_target {
            return target.write().purge_identity(id);
        }
        let Some(_mutation) = self.pin_mutation() else {
            return false;
        };
        let Some(history) = self.doc_lengths.remove(&id) else {
            return false;
        };

        let mut exact = true;
        for entry in &history {
            exact &= self.subtract_aggregate_contribution(
                entry.created_epoch,
                entry.created_by,
                i64::from(entry.len),
                1,
            );
            if let Some(deleted_epoch) = entry.deleted_epoch {
                exact &= self.subtract_aggregate_contribution(
                    deleted_epoch,
                    entry.deleted_by,
                    -i64::from(entry.len),
                    -1,
                );
            }
        }

        self.postings.retain(|_, list| {
            list.postings.retain(|posting| posting.node_id != id);
            !list.postings.is_empty()
        });
        exact
    }

    /// Subtracts one identity-local delta from a matching aggregate bucket.
    /// Each write currently appends its own bucket; accepting a larger bucket
    /// also keeps this exact if a future implementation coalesces equal cuts.
    #[cfg(feature = "lpg")]
    fn subtract_aggregate_contribution(
        &mut self,
        epoch: EpochId,
        tx: Option<TransactionId>,
        total_len: i64,
        doc_count: i64,
    ) -> bool {
        let Some(position) = self.agg_log.iter().position(|delta| {
            delta.epoch == epoch
                && delta.tx == tx
                && delta.d_total_len.signum() == total_len.signum()
                && delta.d_doc_count.signum() == doc_count.signum()
                && delta.d_total_len.unsigned_abs() >= total_len.unsigned_abs()
                && delta.d_doc_count.unsigned_abs() >= doc_count.unsigned_abs()
        }) else {
            return false;
        };
        let delta = &mut self.agg_log[position];
        delta.d_total_len -= total_len;
        delta.d_doc_count -= doc_count;
        if delta.d_total_len == 0 && delta.d_doc_count == 0 {
            self.agg_log.remove(position);
        }
        true
    }

    // ── As-of-epoch aggregate queries ──────────────────────────────────────

    /// Earliest epoch for which this index retains authoritative history.
    #[must_use]
    pub fn retained_from(&self) -> EpochId {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().retained_from();
        }
        self.retained_from
    }

    fn qualify_epoch(&self, epoch: EpochId) -> Result<()> {
        if epoch < self.retained_from {
            return Err(Error::InvalidValue(format!(
                "Text history at epoch {epoch} is unavailable; retained from {}",
                self.retained_from,
            )));
        }
        Ok(())
    }

    /// Returns the total token length visible at `(viewing_epoch, viewing_tx)`.
    ///
    /// # Errors
    /// Rejects epochs below the retained floor.
    pub fn total_length_at(
        &self,
        viewing_epoch: EpochId,
        viewing_tx: TransactionId,
    ) -> Result<u64> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().total_length_at(viewing_epoch, viewing_tx);
        }
        self.qualify_epoch(viewing_epoch)?;
        Ok(self.total_length_qualified(viewing_epoch, viewing_tx))
    }

    fn total_length_qualified(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> u64 {
        let sum: i64 = self
            .agg_log
            .iter()
            .filter(|d| d.visible_to(viewing_epoch, viewing_tx))
            .map(|d| d.d_total_len)
            .sum();
        sum.max(0).cast_unsigned()
    }

    /// Returns the number of documents visible at `(viewing_epoch, viewing_tx)`.
    ///
    /// # Errors
    /// Rejects epochs below the retained floor.
    pub fn doc_count_at(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> Result<u64> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().doc_count_at(viewing_epoch, viewing_tx);
        }
        self.qualify_epoch(viewing_epoch)?;
        Ok(self.doc_count_qualified(viewing_epoch, viewing_tx))
    }

    fn doc_count_qualified(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> u64 {
        let count: i64 = self
            .agg_log
            .iter()
            .filter(|d| d.visible_to(viewing_epoch, viewing_tx))
            .map(|d| d.d_doc_count)
            .sum();
        count.max(0).cast_unsigned()
    }

    /// Returns the average document length visible at `(viewing_epoch, viewing_tx)`.
    /// An empty visible corpus has average length zero.
    ///
    /// # Errors
    /// Rejects epochs below the retained floor.
    pub fn avgdl_at(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> Result<f64> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().avgdl_at(viewing_epoch, viewing_tx);
        }
        self.qualify_epoch(viewing_epoch)?;
        Ok(self.avgdl_qualified(viewing_epoch, viewing_tx))
    }

    fn avgdl_qualified(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> f64 {
        let n = self.doc_count_qualified(viewing_epoch, viewing_tx);
        if n == 0 {
            0.0
        } else {
            self.total_length_qualified(viewing_epoch, viewing_tx) as f64 / n as f64
        }
    }

    /// Returns the doc length of `id` as seen at `(viewing_epoch, viewing_tx)`,
    /// or `None` if the document is not visible.
    fn doc_len_at(
        &self,
        id: NodeId,
        viewing_epoch: EpochId,
        viewing_tx: TransactionId,
    ) -> Option<u32> {
        self.doc_lengths.get(&id)?.iter().find_map(|d| {
            if doc_len_visible(d, viewing_epoch, viewing_tx) {
                Some(d.len)
            } else {
                None
            }
        })
    }

    // ── Garbage collection ─────────────────────────────────────────────────

    /// Garbage collects versioned postings and aggregate-log entries that are
    /// no longer needed by any active snapshot.
    ///
    /// # What is collected
    ///
    /// A `VersionedPosting` is dead weight once its `deleted_epoch` is a
    /// committed value (not `PENDING`) that is **at or below `horizon`**: the
    /// deletion committed before any live transaction could have started, so no
    /// reader will ever ask "was this doc visible at an epoch before its
    /// deletion?"
    ///
    /// Concretely, a posting is removed when:
    ///   `deleted_epoch == Some(d)` where `d != EpochId::PENDING`
    ///     **and** `d.as_u64() <= horizon.as_u64()`
    ///
    /// Live postings (`deleted_epoch == None`) and postings deleted **above**
    /// the horizon are always retained.
    ///
    /// The same rule applies to `VersionedDocLen` entries.
    ///
    /// Empty posting lists and `doc_lengths` entries are pruned after removal.
    ///
    /// # Aggregate log compaction
    ///
    /// All `AggDelta` entries with `epoch <= horizon` and `tx == None`
    /// (committed, no pending tx) are folded into a single base entry stamped
    /// at `horizon` (or `EpochId::new(0)` to remain visible to all epochs >=
    /// horizon).  Entries above the horizon or with a pending `tx` are kept
    /// verbatim.  The prefix-sum invariant is preserved: for any `E >= horizon`
    /// the new log yields the same `total_length_at(E)` / `doc_count_at(E)`.
    ///
    /// Pending (`tx == Some(...)`) entries are never touched — they are owned
    /// by in-flight transactions.
    ///
    /// # Errors
    /// Rejects PENDING horizons, denied mutation authority, or invalid aggregate
    /// arithmetic before changing histories or the monotonic retained floor.
    pub fn gc(&mut self, horizon: EpochId) -> Result<()> {
        if horizon == EpochId::PENDING {
            return Err(Error::InvalidValue(
                "Text GC horizon cannot be PENDING".into(),
            ));
        }
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.write().gc(horizon);
        }
        let _mutation = self.pin_mutation().ok_or(TransactionError::ReadOnly)?;
        let horizon = horizon.max(self.retained_from);
        let mut base_total = 0_i64;
        let mut base_count = 0_i64;
        for delta in &self.agg_log {
            if delta.tx.is_none() && delta.epoch <= horizon {
                base_total = base_total.checked_add(delta.d_total_len).ok_or_else(|| {
                    Error::InvalidValue("Text GC aggregate length overflows".into())
                })?;
                base_count = base_count.checked_add(delta.d_doc_count).ok_or_else(|| {
                    Error::InvalidValue("Text GC aggregate count overflows".into())
                })?;
            }
        }
        if base_total < 0 || base_count < 0 {
            return Err(Error::InvalidValue("Text GC aggregate is negative".into()));
        }

        self.postings.retain(|_, list| {
            list.postings.retain(|posting| {
                posting
                    .deleted_epoch
                    .is_none_or(|deleted| deleted > horizon)
            });
            !list.postings.is_empty()
        });
        self.doc_lengths.retain(|_, history| {
            history.retain(|entry| entry.deleted_epoch.is_none_or(|deleted| deleted > horizon));
            !history.is_empty()
        });
        self.agg_log
            .retain(|delta| delta.tx.is_some() || delta.epoch > horizon);
        if base_total != 0 || base_count != 0 {
            // A nonzero baseline replaces at least one removed entry, so the
            // existing allocation has space. Pending entries retain their order.
            self.agg_log.insert(
                0,
                AggDelta {
                    epoch: EpochId::INITIAL,
                    tx: None,
                    d_total_len: base_total,
                    d_doc_count: base_count,
                },
            );
        }
        self.retained_from = horizon;
        Ok(())
    }

    // ── Legacy (behavior-preserving) wrappers ──────────────────────────────

    /// Indexes a document (node text) into the inverted index.
    ///
    /// If the node was already indexed, it is first removed and re-indexed.
    /// Uses epoch 0 so postings are visible to all committed-latest searches.
    pub fn insert(&mut self, id: NodeId, text: &str) {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            target.write().insert(id, text);
            return;
        }
        let Some(mutation) = self.pin_mutation() else {
            return;
        };
        self.insert_versioned_inner(id, text, EpochId::new(0), None, &mutation);
    }

    /// Removes a document from the index.
    ///
    /// Returns `true` if the document was found and removed.
    /// Uses epoch 0 so the deletion is visible to all committed-latest searches.
    pub fn remove(&mut self, id: NodeId) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.write().remove(id);
        }
        let Some(mutation) = self.pin_mutation() else {
            return false;
        };
        self.remove_versioned_inner(id, EpochId::new(0), None, &mutation)
    }

    // ── BM25 search ────────────────────────────────────────────────────────

    /// BM25 term score: IDF * TF-component for a single term occurrence.
    ///
    /// `df` is the document frequency (number of documents containing the term),
    /// `tf` is the term frequency in this document, `dl` is the document length,
    /// `n` is the corpus size, and `avg_dl` is the average document length.
    #[inline]
    fn bm25_term_score(&self, df: f64, tf: f64, dl: f64, n: f64, avg_dl: f64) -> f64 {
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        let tf_component = (tf * (self.config.k1 + 1.0))
            / (tf + self.config.k1 * (1.0 - self.config.b + self.config.b * dl / avg_dl));
        idf * tf_component
    }

    /// Searches the index using BM25 scoring.
    ///
    /// Returns up to `k` results sorted by descending BM25 score.
    /// Uses the committed-latest view (all epoch-0 inserts, no pending deletes).
    pub fn search(&self, query: &str, k: usize) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().search(query, k);
        }
        let query_tokens = self.tokenizer.tokenize(query);
        let n = self.doc_count_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        if query_tokens.is_empty() || n == 0 {
            return Vec::new();
        }

        let n_f = n as f64;
        let avg_dl = self.avgdl_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        let mut scores: HashMap<NodeId, f64> = HashMap::new();

        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            // Count only the visible postings for df.
            let df = posting_list
                .postings
                .iter()
                .filter(|p| posting_visible(p, COMMITTED_EPOCH, TransactionId::INVALID))
                .count() as f64;
            if df == 0.0 {
                continue;
            }
            for posting in &posting_list.postings {
                if !posting_visible(posting, COMMITTED_EPOCH, TransactionId::INVALID) {
                    continue;
                }
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(
                    self.doc_len_at(posting.node_id, COMMITTED_EPOCH, TransactionId::INVALID)
                        .unwrap_or(0),
                );
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n_f, avg_dl);
            }
        }

        let mut results: Vec<(NodeId, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Scores a single document against a query using BM25.
    ///
    /// Looks up each query term in its posting list, finds the entry for the
    /// given node ID, and computes BM25 with corpus statistics. Returns `0.0`
    /// if the document has no matching terms or doesn't exist.
    ///
    /// Cost is O(query_terms × average posting-list length) per call: for each
    /// query term, the matching node is found by linear scan of that term's
    /// posting list. Intended for per-row evaluation, where a few hundred
    /// per-document scores are cheaper than reorganizing posting lists into
    /// per-document maps.
    #[must_use]
    pub fn score_document(&self, id: NodeId, query: &str) -> f64 {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().score_document(id, query);
        }
        let query_tokens = self.tokenizer.tokenize(query);
        let n = self.doc_count_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        if query_tokens.is_empty() || n == 0 {
            return 0.0;
        }
        let Some(doc_len) = self.doc_len_at(id, COMMITTED_EPOCH, TransactionId::INVALID) else {
            return 0.0;
        };
        let n_f = n as f64;
        let avg_dl = self.avgdl_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        let dl = f64::from(doc_len);
        let mut score = 0.0;
        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = posting_list
                .postings
                .iter()
                .filter(|p| posting_visible(p, COMMITTED_EPOCH, TransactionId::INVALID))
                .count() as f64;
            if df == 0.0 {
                continue;
            }
            let tf = posting_list
                .postings
                .iter()
                .find(|p| {
                    p.node_id == id && posting_visible(p, COMMITTED_EPOCH, TransactionId::INVALID)
                })
                .map_or(0.0, |p| f64::from(p.term_freq));
            if tf > 0.0 {
                score += self.bm25_term_score(df, tf, dl, n_f, avg_dl);
            }
        }
        score
    }

    /// Scores a single document against a query at a specific `(epoch, tx)` snapshot.
    ///
    /// Like [`Self::score_document`] but filters each posting list by
    /// `posting_visible(epoch, tx)` and uses `avgdl_at(epoch, tx)` so the
    /// score reflects only what was visible to the reader at snapshot time.
    ///
    /// A delta doc (an uncommitted insert for `node_id` in `tx`) can be supplied
    /// via `delta_doc`: if `Some((doc_len, freq_map))` the committed posting for
    /// `node_id` is ignored and the delta values are used instead; if the delta
    /// signals a tombstone (`delta_removed == true`) the function returns `None`
    /// immediately (the node is not visible).
    ///
    /// Returns `None` when the node has no visible entry in this index or the
    /// query has no tokens. A visible document with no matching query terms
    /// returns `Some(0.0)`, not an absent score.
    ///
    /// # Errors
    /// Rejects epochs below the retained floor, even for tombstones or empty queries.
    pub fn score_document_visible(
        &self,
        id: NodeId,
        query: &str,
        epoch: EpochId,
        tx: TransactionId,
        delta_doc: Option<(u32, &HashMap<String, u32>)>,
        delta_removed: bool,
    ) -> Result<Option<f64>> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().score_document_visible(
                id,
                query,
                epoch,
                tx,
                delta_doc,
                delta_removed,
            );
        }
        self.qualify_epoch(epoch)?;
        // A tombstone in this tx means the node is invisible.
        if delta_removed {
            return Ok(None);
        }

        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() {
            return Ok(None);
        }

        // Effective doc length for `id`: delta overrides committed.
        let dl = if let Some((delta_len, _)) = delta_doc {
            f64::from(delta_len)
        } else {
            let Some(length) = self.doc_len_at(id, epoch, tx) else {
                return Ok(None);
            };
            f64::from(length)
        };

        // Corpus stats at (epoch, tx).
        let n = self.doc_count_qualified(epoch, tx);
        if n == 0 {
            return Ok(None);
        }
        let n_f = n as f64;
        let avg_dl = self.avgdl_qualified(epoch, tx);
        let avg_dl = if avg_dl <= 0.0 { 1.0 } else { avg_dl };

        let mut score = 0.0;
        for token in &query_tokens {
            // df: committed visible postings for this term (excluding `id` if delta
            // overrides it), plus 1 if the delta doc contains this token.
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                // Term not in committed index; only delta could contribute.
                if let Some((_, freq_map)) = delta_doc
                    && let Some(&tf) = freq_map.get(token.as_str())
                {
                    // df = 1 (only this delta doc); score it against corpus stats.
                    let df = 1.0_f64;
                    score += self.bm25_term_score(df, f64::from(tf), dl, n_f, avg_dl);
                }
                continue;
            };

            let committed_df = posting_list
                .postings
                .iter()
                .filter(|p| {
                    posting_visible(p, epoch, tx)
                        // If `id` is covered by a delta doc, exclude its committed posting
                        // from df (it will be re-counted via the delta path below).
                        && !(delta_doc.is_some() && p.node_id == id)
                })
                .count();

            let delta_has_term = delta_doc
                .and_then(|(_, fm)| fm.get(token.as_str()))
                .is_some();

            let df = (committed_df + usize::from(delta_has_term)) as f64;
            if df == 0.0 {
                continue;
            }

            // tf: delta overrides committed for `id`.
            let tf = if let Some((_, freq_map)) = delta_doc {
                f64::from(freq_map.get(token.as_str()).copied().unwrap_or(0))
            } else {
                posting_list
                    .postings
                    .iter()
                    .find(|p| p.node_id == id && posting_visible(p, epoch, tx))
                    .map_or(0.0, |p| f64::from(p.term_freq))
            };

            if tf > 0.0 {
                score += self.bm25_term_score(df, tf, dl, n_f, avg_dl);
            }
        }

        // Visibility was qualified above; a nonmatch is a score, not absence.
        Ok(Some(score))
    }

    /// Returns all documents scoring at or above `threshold` using BM25.
    ///
    /// Unlike [`Self::search`] (top-k), this returns every document above the
    /// threshold, sorted by score descending. Intended for index-accelerated
    /// text search with WHERE predicates.
    #[must_use]
    pub fn search_with_threshold(&self, query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().search_with_threshold(query, threshold);
        }
        let query_tokens = self.tokenizer.tokenize(query);
        let n = self.doc_count_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        if query_tokens.is_empty() || n == 0 {
            return Vec::new();
        }
        let n_f = n as f64;
        let avg_dl = self.avgdl_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = posting_list
                .postings
                .iter()
                .filter(|p| posting_visible(p, COMMITTED_EPOCH, TransactionId::INVALID))
                .count() as f64;
            if df == 0.0 {
                continue;
            }
            for posting in &posting_list.postings {
                if !posting_visible(posting, COMMITTED_EPOCH, TransactionId::INVALID) {
                    continue;
                }
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(
                    self.doc_len_at(posting.node_id, COMMITTED_EPOCH, TransactionId::INVALID)
                        .unwrap_or(0),
                );
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n_f, avg_dl);
            }
        }
        let mut results: Vec<(NodeId, f64)> = scores
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // ── Snapshot threshold search ───────────────────────────────────────────

    /// Searches the index at a specific `(epoch, tx)` snapshot, returning every
    /// document whose BM25 score meets or exceeds `threshold`.
    ///
    /// Mirrors [`Self::search_visible`] but uses a threshold cutoff instead of a top-k
    /// limit.  Committed postings are filtered by `posting_visible(epoch, tx)`;
    /// the per-transaction delta is merged so a writer sees its own uncommitted
    /// inserts/tombstones.
    ///
    /// # Parameters
    ///
    /// Same as [`Self::search_visible`] except `threshold` replaces `k`.
    ///
    /// # Errors
    /// Rejects epochs below the retained floor, including empty queries.
    pub fn search_with_threshold_visible(
        &self,
        query: &str,
        threshold: f64,
        epoch: EpochId,
        tx: TransactionId,
        delta_docs: &[(NodeId, String)],
        delta_removed: &FxHashSet<NodeId>,
    ) -> Result<Vec<(NodeId, f64)>> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().search_with_threshold_visible(
                query,
                threshold,
                epoch,
                tx,
                delta_docs,
                delta_removed,
            );
        }
        self.qualify_epoch(epoch)?;
        let query_tokens = self.tokenizer.tokenize(query);
        // An empty query or empty corpus never yields results.
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Build per-delta-doc token maps (identical to search_visible step 1).
        let delta_token_maps: Vec<(NodeId, HashMap<String, u32>, u32)> = delta_docs
            .iter()
            .map(|(node_id, text)| {
                let tokens = self.tokenizer.tokenize(text);
                #[allow(clippy::cast_possible_truncation)]
                let doc_len = tokens.len() as u32;
                let mut freq_map: HashMap<String, u32> = HashMap::new();
                for t in tokens {
                    *freq_map.entry(t).or_insert(0) += 1;
                }
                (*node_id, freq_map, doc_len)
            })
            .collect();

        // Corpus stats (identical to search_visible step 2).
        #[allow(clippy::cast_possible_wrap)]
        let base_n = self.doc_count_qualified(epoch, tx) as i64;
        #[allow(clippy::cast_possible_wrap)]
        let base_total = self.total_length_qualified(epoch, tx) as i64;

        let mut len_adjustment: i64 = 0;
        let mut count_adjustment: i64 = 0;

        for (node_id, _, delta_len) in &delta_token_maps {
            let committed_len = self.doc_len_at(*node_id, epoch, tx);
            if let Some(cl) = committed_len {
                len_adjustment += i64::from(*delta_len) - i64::from(cl);
            } else {
                len_adjustment += i64::from(*delta_len);
                count_adjustment += 1;
            }
        }

        let delta_doc_nodes: FxHashSet<NodeId> =
            delta_token_maps.iter().map(|(n, _, _)| *n).collect();
        for &removed_node in delta_removed {
            if !delta_doc_nodes.contains(&removed_node)
                && self.doc_len_at(removed_node, epoch, tx).is_some()
            {
                let cl = self.doc_len_at(removed_node, epoch, tx).unwrap_or(0);
                len_adjustment -= i64::from(cl);
                count_adjustment -= 1;
            }
        }

        let n_eff = (base_n + count_adjustment).max(1) as f64;
        let total_eff = (base_total + len_adjustment).max(0) as f64;
        let avg_dl_eff = total_eff / n_eff;
        let avg_dl = if avg_dl_eff <= 0.0 { 1.0 } else { avg_dl_eff };

        // Score candidates (same shape as search_visible step 3, no truncation).
        let mut scores: HashMap<NodeId, f64> = HashMap::new();

        for token in &query_tokens {
            let committed_visible_for_term: Vec<&super::versioned::VersionedPosting> = self
                .postings
                .get(token.as_str())
                .map(|pl| {
                    pl.postings
                        .iter()
                        .filter(|p| {
                            posting_visible(p, epoch, tx)
                                && !delta_removed.contains(&p.node_id)
                                && !delta_doc_nodes.contains(&p.node_id)
                        })
                        .collect()
                })
                .unwrap_or_default();

            let delta_hits: Vec<(&NodeId, u32, u32)> = delta_token_maps
                .iter()
                .filter_map(|(nid, freq_map, dl)| {
                    freq_map.get(token.as_str()).map(|&tf| (nid, tf, *dl))
                })
                .collect();

            let df = (committed_visible_for_term.len() + delta_hits.len()) as f64;
            if df == 0.0 {
                continue;
            }

            for posting in committed_visible_for_term {
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(self.doc_len_at(posting.node_id, epoch, tx).unwrap_or(0));
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n_eff, avg_dl);
            }

            for (nid, tf, dl) in &delta_hits {
                let tf_f = f64::from(*tf);
                let dl_f = f64::from(*dl);
                *scores.entry(**nid).or_insert(0.0) +=
                    self.bm25_term_score(df, tf_f, dl_f, n_eff, avg_dl);
            }
        }

        let mut results: Vec<(NodeId, f64)> = scores
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    // ── Snapshot search (TI4) ──────────────────────────────────────────────

    /// Searches the index at a specific `(epoch, tx)` snapshot, merging the
    /// committed postings with an in-flight transactional delta.
    ///
    /// # Parameters
    ///
    /// - `query` — raw query text (tokenised internally).
    /// - `k` — maximum results to return.
    /// - `epoch` — the snapshot epoch for committed visibility.
    /// - `tx` — the viewing transaction (own-tx pending postings are visible to
    ///   this tx only; `TransactionId::INVALID` for no pending own-tx).
    /// - `delta_docs` — `(NodeId, text)` pairs the transaction has buffered as
    ///   inserts/updates for this index.  These replace any committed posting for
    ///   the same node.
    /// - `delta_removed` — `NodeId`s the transaction has buffered as tombstones.
    ///   These nodes are always excluded from the result even if they have a live
    ///   committed posting.
    ///
    /// # Corpus-stat treatment
    ///
    /// The base corpus stats (`n`, `avg_dl`) come from `doc_count_at(epoch, tx)` /
    /// `avgdl_at(epoch, tx)`.  Delta docs are small relative to the committed
    /// corpus, so instead of exact bookkeeping we apply a lightweight adjustment:
    ///
    /// - Nodes in `delta_docs` that already had a committed posting are treated as
    ///   *replacements* (their committed length is subtracted and their new delta
    ///   length is added); no net count change for those nodes.
    /// - Nodes in `delta_docs` that had *no* committed posting add 1 to `n`.
    /// - Nodes in `delta_removed` that had a committed posting subtract 1 from `n`
    ///   (only if they are not also in `delta_docs`).
    ///
    /// The **doc set** is exact: delta inserts always appear, delta tombstones
    /// never appear.  The `avg_dl` approximation is minor for small deltas.
    ///
    /// # Errors
    /// Rejects epochs below the retained floor, including empty queries and zero limits.
    pub fn search_visible(
        &self,
        query: &str,
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
        delta_docs: &[(NodeId, String)],
        delta_removed: &FxHashSet<NodeId>,
    ) -> Result<Vec<(NodeId, f64)>> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target
                .read()
                .search_visible(query, k, epoch, tx, delta_docs, delta_removed);
        }
        self.qualify_epoch(epoch)?;
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // ── Step 1: build per-delta-doc token maps ──────────────────────────
        // For each delta doc, tokenize and compute (term_freq_map, doc_len).
        let delta_token_maps: Vec<(NodeId, HashMap<String, u32>, u32)> = delta_docs
            .iter()
            .map(|(node_id, text)| {
                let tokens = self.tokenizer.tokenize(text);
                // reason: token count fits u32 for any practical document
                #[allow(clippy::cast_possible_truncation)]
                let doc_len = tokens.len() as u32;
                let mut freq_map: HashMap<String, u32> = HashMap::new();
                for t in tokens {
                    *freq_map.entry(t).or_insert(0) += 1;
                }
                (*node_id, freq_map, doc_len)
            })
            .collect();

        // ── Step 2: compute effective corpus stats ──────────────────────────
        // Base from committed view at (epoch, tx).
        // reason: doc_count and total_length are bounded by practical corpus sizes
        // and will never exceed i64::MAX; the cast is intentional.
        #[allow(clippy::cast_possible_wrap)]
        let base_n = self.doc_count_qualified(epoch, tx) as i64;
        #[allow(clippy::cast_possible_wrap)]
        let base_total = self.total_length_qualified(epoch, tx) as i64;

        // Partition delta_docs into replacements (committed posting exists) and
        // new inserts (no committed posting exists).
        let mut len_adjustment: i64 = 0;
        let mut count_adjustment: i64 = 0;

        for (node_id, _, delta_len) in &delta_token_maps {
            let committed_len = self.doc_len_at(*node_id, epoch, tx);
            if let Some(cl) = committed_len {
                // Replacement: subtract old length, add new length.
                len_adjustment += i64::from(*delta_len) - i64::from(cl);
                // No count change.
            } else {
                // New insert: add length and count.
                len_adjustment += i64::from(*delta_len);
                count_adjustment += 1;
            }
        }

        // Tombstoned nodes that had a committed posting reduce count, unless they
        // are also in delta_docs (which would be a set-then-remove; the net is
        // "remove", handled by not adding them to delta_token_maps above).
        let delta_doc_nodes: FxHashSet<NodeId> =
            delta_token_maps.iter().map(|(n, _, _)| *n).collect();
        for &removed_node in delta_removed {
            if !delta_doc_nodes.contains(&removed_node)
                && self.doc_len_at(removed_node, epoch, tx).is_some()
            {
                // The doc is being removed; subtract its committed length too.
                let cl = self.doc_len_at(removed_node, epoch, tx).unwrap_or(0);
                len_adjustment -= i64::from(cl);
                count_adjustment -= 1;
            }
        }

        let n_eff = (base_n + count_adjustment).max(1) as f64;
        let total_eff = (base_total + len_adjustment).max(0) as f64;
        let avg_dl_eff = total_eff / n_eff;
        // Avoid division by zero: use 1.0 when corpus is effectively empty.
        let avg_dl = if avg_dl_eff <= 0.0 { 1.0 } else { avg_dl_eff };

        // ── Step 3: score candidates ────────────────────────────────────────
        let mut scores: HashMap<NodeId, f64> = HashMap::new();

        for token in &query_tokens {
            // ── A. Committed postings visible at (epoch, tx), minus tombstones ──
            //
            // Also build the effective df for this term: count committed visible
            // docs (minus tombstones, minus those overridden by delta_docs) plus
            // delta docs that contain this term.
            let committed_visible_for_term: Vec<&super::versioned::VersionedPosting> = self
                .postings
                .get(token.as_str())
                .map(|pl| {
                    pl.postings
                        .iter()
                        .filter(|p| {
                            posting_visible(p, epoch, tx)
                                && !delta_removed.contains(&p.node_id)
                                && !delta_doc_nodes.contains(&p.node_id)
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Delta docs that contain this token.
            let delta_hits: Vec<(&NodeId, u32, u32)> = delta_token_maps
                .iter()
                .filter_map(|(nid, freq_map, dl)| {
                    freq_map.get(token.as_str()).map(|&tf| (nid, tf, *dl))
                })
                .collect();

            let df = (committed_visible_for_term.len() + delta_hits.len()) as f64;
            if df == 0.0 {
                continue;
            }

            // Score committed visible postings.
            for posting in committed_visible_for_term {
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(self.doc_len_at(posting.node_id, epoch, tx).unwrap_or(0));
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n_eff, avg_dl);
            }

            // Score delta inserts.
            for (nid, tf, dl) in &delta_hits {
                let tf_f = f64::from(*tf);
                let dl_f = f64::from(*dl);
                *scores.entry(**nid).or_insert(0.0) +=
                    self.bm25_term_score(df, tf_f, dl_f, n_eff, avg_dl);
            }
        }

        let mut results: Vec<(NodeId, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    // ── Query helpers ───────────────────────────────────────────────────────

    /// Returns true if the given node has a live (committed-latest) entry.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().contains(id);
        }
        self.doc_lengths.get(&id).is_some_and(|v| {
            v.iter()
                .any(|d| doc_len_visible(d, COMMITTED_EPOCH, TransactionId::INVALID))
        })
    }

    /// Returns the number of committed-latest indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().len();
        }
        // reason: practical document counts fit usize on all supported platforms
        #[allow(clippy::cast_possible_truncation)]
        let n = self.doc_count_qualified(COMMITTED_EPOCH, TransactionId::INVALID) as usize;
        n
    }

    /// Returns true if the index is empty (committed-latest view).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().is_empty();
        }
        self.doc_count_qualified(COMMITTED_EPOCH, TransactionId::INVALID) == 0
    }

    /// Returns the number of unique terms in the index.
    #[must_use]
    pub fn term_count(&self) -> usize {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().term_count();
        }
        self.postings.len()
    }

    /// Returns an owned snapshot of the BM25 configuration.
    ///
    /// The owned value remains exact even if a downstream caller moves a
    /// forwarding handle out of its original outer lock: the private target is
    /// read under its lock and no reference escapes after that guard releases.
    #[must_use]
    pub fn config(&self) -> BM25Config {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().config();
        }
        self.config.clone()
    }

    /// Returns whether this index uses the built-in Simple tokenizer with the
    /// exact requested minimum token length. Opaque tokenizers never match.
    #[must_use]
    pub fn has_simple_tokenizer(&self, min_token_length: usize) -> bool {
        u64::try_from(min_token_length).is_ok_and(|min_token_length| {
            self.tokenizer_matches(&ExactTokenizerDescriptor::Simple { min_token_length })
        })
    }

    /// Compares a resolved tokenizer contract without capturing index contents.
    pub(super) fn tokenizer_matches(&self, expected: &ExactTokenizerDescriptor) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().tokenizer_matches(expected);
        }
        self.tokenizer_descriptor.as_ref() == Some(expected)
    }

    /// Whether the current persistence descriptors can reconstruct this index
    /// exactly from the graph's current text values.
    ///
    /// Those descriptors encode BM25 parameters but not tokenizer identity or
    /// MVCC posting history. Therefore only the default tokenizer and a single
    /// live epoch-zero image are representable. Callers must fail closed for
    /// any other state instead of silently changing historical search results.
    #[must_use]
    pub fn is_current_image_persistence_representable(&self) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().is_current_image_persistence_representable();
        }
        if self.tokenizer_descriptor
            != Some(ExactTokenizerDescriptor::Simple {
                min_token_length: 2,
            })
        {
            return false;
        }
        if self.retained_from != EpochId::INITIAL {
            return false;
        }
        if !self
            .agg_log
            .iter()
            .all(|delta| delta.epoch == EpochId::INITIAL && delta.tx.is_none())
        {
            return false;
        }
        if !self.doc_lengths.values().all(|history| {
            matches!(history.as_slice(), [entry]
                if entry.created_epoch == EpochId::INITIAL
                    && entry.created_by.is_none()
                    && entry.deleted_epoch.is_none()
                    && entry.deleted_by.is_none())
        }) {
            return false;
        }
        self.postings.values().all(|list| {
            list.postings.iter().all(|posting| {
                posting.created_epoch == EpochId::INITIAL
                    && posting.created_by.is_none()
                    && posting.deleted_epoch.is_none()
                    && posting.deleted_by.is_none()
            })
        })
    }

    // ── Snapshot / restore ─────────────────────────────────────────────────

    /// Captures every committed posting, document-length record and aggregate
    /// delta without flattening MVCC history.
    ///
    /// Opaque custom tokenizers and any pending transaction state fail closed.
    /// This is a temporary codec boundary, not permission to substitute the
    /// default tokenizer or discard historical state.
    pub(super) fn exact_committed_image(&self) -> std::result::Result<ExactTextIndexImage, String> {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().exact_committed_image();
        }

        let Some(tokenizer) = self.tokenizer_descriptor.clone() else {
            return Err(
                "opaque custom tokenizer has no stable persistence codec; exact serialization is temporarily unavailable and no fallback was used"
                    .to_string(),
            );
        };

        let mut postings = self
            .postings
            .iter()
            .map(|(term, list)| ExactPostingList {
                term: term.clone(),
                postings: list
                    .postings
                    .iter()
                    .map(|posting| ExactPosting {
                        node_id: posting.node_id,
                        term_freq: posting.term_freq,
                        created_epoch: posting.created_epoch,
                        created_by: posting.created_by,
                        deleted_epoch: posting.deleted_epoch,
                        deleted_by: posting.deleted_by,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        postings.sort_by(|left, right| left.term.cmp(&right.term));

        let mut doc_lengths = self
            .doc_lengths
            .iter()
            .map(|(node_id, history)| ExactDocHistory {
                node_id: *node_id,
                history: history
                    .iter()
                    .map(|entry| ExactDocLength {
                        len: entry.len,
                        created_epoch: entry.created_epoch,
                        created_by: entry.created_by,
                        deleted_epoch: entry.deleted_epoch,
                        deleted_by: entry.deleted_by,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        doc_lengths.sort_by_key(|history| history.node_id);

        let image = ExactTextIndexImage {
            k1: self.config.k1,
            b: self.config.b,
            tokenizer,
            retained_from: self.retained_from,
            postings,
            doc_lengths,
            agg_log: self
                .agg_log
                .iter()
                .map(|delta| ExactAggDelta {
                    epoch: delta.epoch,
                    tx: delta.tx,
                    d_total_len: delta.d_total_len,
                    d_doc_count: delta.d_doc_count,
                })
                .collect(),
        };
        validate_exact_image(&image)?;
        Ok(image)
    }

    /// Validates and materializes a current image without touching a live index.
    pub(super) fn prepare_exact_image(
        image: ExactTextIndexImage,
    ) -> std::result::Result<PreparedTextIndexImage, String> {
        validate_exact_image(&image)?;

        let tokenizer: Arc<dyn Tokenizer> = match &image.tokenizer {
            ExactTokenizerDescriptor::Simple { min_token_length } => {
                let min_token_length = usize::try_from(*min_token_length).map_err(|_| {
                    format!(
                        "SimpleTokenizer minimum length {min_token_length} does not fit this target"
                    )
                })?;
                Arc::new(SimpleTokenizer::with_min_length(min_token_length))
            }
        };

        let postings = image
            .postings
            .into_iter()
            .map(|list| {
                (
                    list.term,
                    PostingList {
                        postings: list
                            .postings
                            .into_iter()
                            .map(|posting| VersionedPosting {
                                node_id: posting.node_id,
                                term_freq: posting.term_freq,
                                created_epoch: posting.created_epoch,
                                created_by: posting.created_by,
                                deleted_epoch: posting.deleted_epoch,
                                deleted_by: posting.deleted_by,
                            })
                            .collect(),
                    },
                )
            })
            .collect();
        let doc_lengths = image
            .doc_lengths
            .into_iter()
            .map(|history| {
                (
                    history.node_id,
                    history
                        .history
                        .into_iter()
                        .map(|entry| VersionedDocLen {
                            len: entry.len,
                            created_epoch: entry.created_epoch,
                            created_by: entry.created_by,
                            deleted_epoch: entry.deleted_epoch,
                            deleted_by: entry.deleted_by,
                        })
                        .collect(),
                )
            })
            .collect();
        let agg_log = image
            .agg_log
            .into_iter()
            .map(|delta| AggDelta {
                epoch: delta.epoch,
                tx: delta.tx,
                d_total_len: delta.d_total_len,
                d_doc_count: delta.d_doc_count,
            })
            .collect();

        Ok(PreparedTextIndexImage {
            config: BM25Config {
                k1: image.k1,
                b: image.b,
            },
            tokenizer,
            tokenizer_descriptor: image.tokenizer,
            retained_from: image.retained_from,
            postings,
            doc_lengths,
            agg_log,
        })
    }

    /// Retains the exact scope proof used by a prepared restore transaction.
    pub(super) fn pin_prepared_restore(&self) -> Option<TextMutation> {
        self.pin_mutation()
    }

    /// Installs a fully validated image with no remaining fallible work.
    pub(super) fn install_prepared_image(
        &mut self,
        prepared: PreparedTextIndexImage,
        _mutation: &TextMutation,
    ) {
        #[cfg(feature = "lpg")]
        debug_assert!(
            self.registry_target.is_none(),
            "prepared restore must target a concrete registry index"
        );
        self.config = prepared.config;
        self.tokenizer = prepared.tokenizer;
        self.tokenizer_descriptor = Some(prepared.tokenizer_descriptor);
        self.retained_from = prepared.retained_from;
        self.postings = prepared.postings;
        self.doc_lengths = prepared.doc_lengths;
        self.agg_log = prepared.agg_log;
    }

    /// Snapshot the index for serialization.
    ///
    /// Returns (postings, doc_lengths, total_length) where postings is
    /// a vec of (term, vec of (node_id, term_freq)).
    ///
    /// Only committed-latest-visible postings are included in the snapshot.
    #[must_use]
    pub fn snapshot(&self) -> (Vec<(String, Vec<(NodeId, u32)>)>, Vec<(NodeId, u32)>, u64) {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().snapshot();
        }
        let mut postings: Vec<(String, Vec<(NodeId, u32)>)> = self
            .postings
            .iter()
            .map(|(term, pl)| {
                let entries: Vec<(NodeId, u32)> = pl
                    .postings
                    .iter()
                    .filter(|p| posting_visible(p, COMMITTED_EPOCH, TransactionId::INVALID))
                    .map(|p| (p.node_id, p.term_freq))
                    .collect();
                (term.clone(), entries)
            })
            .filter(|(_, entries)| !entries.is_empty())
            .collect();
        postings.sort_by(|(a, _), (b, _)| a.cmp(b));

        // Snapshot the committed-latest doc lengths.
        let mut doc_lengths: Vec<(NodeId, u32)> = self
            .doc_lengths
            .iter()
            .filter_map(|(id, history)| {
                history
                    .iter()
                    .find(|d| doc_len_visible(d, COMMITTED_EPOCH, TransactionId::INVALID))
                    .map(|d| (*id, d.len))
            })
            .collect();
        doc_lengths.sort_by_key(|(id, _)| *id);

        let total_length = self.total_length_qualified(COMMITTED_EPOCH, TransactionId::INVALID);
        (postings, doc_lengths, total_length)
    }

    /// Override the BM25 configuration parameters.
    pub fn set_config(&mut self, config: BM25Config) {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            target.write().set_config(config);
            return;
        }
        let Some(_mutation) = self.pin_mutation() else {
            return;
        };
        self.config = config;
    }

    /// Restore the index from a snapshot. Replaces all current data.
    ///
    /// Restored postings and doc lengths are stamped with epoch 0 (always-visible).
    pub fn restore(
        &mut self,
        postings: Vec<(String, Vec<(NodeId, u32)>)>,
        doc_lengths: Vec<(NodeId, u32)>,
        total_length: u64,
    ) {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            target.write().restore(postings, doc_lengths, total_length);
            return;
        }
        let Some(mutation) = self.pin_mutation() else {
            return;
        };
        self.restore_inner(postings, doc_lengths, total_length, &mutation);
    }

    fn restore_inner(
        &mut self,
        postings: Vec<(String, Vec<(NodeId, u32)>)>,
        doc_lengths: Vec<(NodeId, u32)>,
        total_length: u64,
        _mutation: &TextMutation,
    ) {
        self.postings.clear();
        for (term, entries) in postings {
            let posting_list = PostingList {
                postings: entries
                    .into_iter()
                    .map(|(node_id, term_freq)| {
                        VersionedPosting::new(node_id, term_freq, EpochId::new(0), None)
                    })
                    .collect(),
            };
            self.postings.insert(term, posting_list);
        }
        // Rebuild versioned doc_lengths and agg_log from the flat snapshot.
        self.doc_lengths.clear();
        self.agg_log.clear();
        for (id, len) in doc_lengths {
            self.doc_lengths
                .entry(id)
                .or_default()
                .push(VersionedDocLen::new(len, EpochId::new(0), None));
            self.agg_log.push(AggDelta {
                epoch: EpochId::new(0),
                tx: None,
                d_total_len: i64::from(len),
                d_doc_count: 1,
            });
        }
        // Sanity: the computed total should match what was snapshotted.
        let _ = total_length; // used only as a cross-check during debugging
    }

    /// Returns estimated heap memory in bytes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        #[cfg(feature = "lpg")]
        if let Some(target) = &self.registry_target {
            return target.read().heap_memory_bytes();
        }
        // Postings map: term strings + PostingList vecs
        let postings_overhead = self.postings.capacity()
            * (std::mem::size_of::<String>() + std::mem::size_of::<PostingList>() + 1);
        let postings_data: usize = self
            .postings
            .iter()
            .map(|(term, pl)| {
                term.len() + pl.postings.capacity() * std::mem::size_of::<VersionedPosting>()
            })
            .sum();
        // Doc lengths map: NodeId → Vec<VersionedDocLen>
        let doc_lengths_bytes: usize = self
            .doc_lengths
            .values()
            .map(|history| {
                std::mem::size_of::<NodeId>()
                    + history.capacity() * std::mem::size_of::<VersionedDocLen>()
            })
            .sum();
        // Aggregate log
        let agg_log_bytes = self.agg_log.capacity() * std::mem::size_of::<AggDelta>();
        postings_overhead + postings_data + doc_lengths_bytes + agg_log_bytes
    }
}

type ExactLifecycle = (NodeId, u64, Option<u64>);

fn validate_committed_event(
    epoch: EpochId,
    tx: Option<TransactionId>,
    context: &str,
) -> std::result::Result<(), String> {
    if epoch == EpochId::PENDING {
        let Some(tx) = tx else {
            return Err(format!(
                "{context} uses PENDING epoch without an owning transaction"
            ));
        };
        if !tx.is_valid() {
            return Err(format!("{context} uses the invalid transaction ID"));
        }
        return Err(format!(
            "{context} contains pending transaction state for transaction {tx}"
        ));
    }
    if let Some(tx) = tx {
        if !tx.is_valid() {
            return Err(format!("{context} uses the invalid transaction ID"));
        }
        return Err(format!(
            "{context} carries transaction {tx} with committed epoch {epoch}"
        ));
    }
    Ok(())
}

fn validate_committed_lifecycle(
    created_epoch: EpochId,
    created_by: Option<TransactionId>,
    deleted_epoch: Option<EpochId>,
    deleted_by: Option<TransactionId>,
    context: &str,
) -> std::result::Result<(), String> {
    validate_committed_event(created_epoch, created_by, &format!("{context} creation"))?;
    match deleted_epoch {
        Some(deleted_epoch) => {
            validate_committed_event(deleted_epoch, deleted_by, &format!("{context} deletion"))?;
            if deleted_epoch.as_u64() < created_epoch.as_u64() {
                return Err(format!(
                    "{context} is deleted at epoch {deleted_epoch} before creation at epoch {created_epoch}"
                ));
            }
        }
        None if deleted_by.is_some() => {
            return Err(format!(
                "{context} carries a deleting transaction without a deletion epoch"
            ));
        }
        None => {}
    }
    Ok(())
}

fn validate_history_order(
    previous: Option<(u64, Option<u64>)>,
    created_epoch: EpochId,
    context: &str,
) -> std::result::Result<(), String> {
    let Some((previous_created, previous_deleted)) = previous else {
        return Ok(());
    };
    if created_epoch.as_u64() < previous_created {
        return Err(format!("{context} is not ordered by creation epoch"));
    }
    let Some(previous_deleted) = previous_deleted else {
        return Err(format!("{context} contains overlapping live histories"));
    };
    if previous_deleted > created_epoch.as_u64() {
        return Err(format!("{context} contains overlapping lifetimes"));
    }
    Ok(())
}

fn validate_exact_image(image: &ExactTextIndexImage) -> std::result::Result<(), String> {
    if image.retained_from == EpochId::PENDING {
        return Err("text retained floor cannot be PENDING".into());
    }
    match image.tokenizer {
        ExactTokenizerDescriptor::Simple { min_token_length } => {
            usize::try_from(min_token_length).map_err(|_| {
                format!(
                    "SimpleTokenizer minimum length {min_token_length} does not fit this target"
                )
            })?;
        }
    }

    let mut node_ids = HashSet::with_capacity(image.doc_lengths.len());
    let mut doc_versions: HashMap<ExactLifecycle, Vec<u32>> = HashMap::new();
    let mut live_doc_count = 0_u64;
    let mut live_total_length = 0_u64;

    for doc in &image.doc_lengths {
        if !doc.node_id.is_valid() {
            return Err("text index contains the invalid node ID".to_string());
        }
        if !node_ids.insert(doc.node_id) {
            return Err(format!(
                "text index contains duplicate document-history ID {}",
                doc.node_id
            ));
        }
        if doc.history.is_empty() {
            return Err(format!(
                "document {} has an empty length history",
                doc.node_id
            ));
        }

        let mut previous = None;
        for (position, entry) in doc.history.iter().enumerate() {
            let context = format!("document {} history record {position}", doc.node_id);
            if entry.len == 0 {
                return Err(format!("{context} has zero token length"));
            }
            validate_committed_lifecycle(
                entry.created_epoch,
                entry.created_by,
                entry.deleted_epoch,
                entry.deleted_by,
                &context,
            )?;
            if image.retained_from != EpochId::INITIAL
                && entry
                    .deleted_epoch
                    .is_some_and(|deleted| deleted <= image.retained_from)
            {
                return Err(format!("{context} ends at or below the retained floor"));
            }
            validate_history_order(
                previous,
                entry.created_epoch,
                &format!("document {} history", doc.node_id),
            )?;
            previous = Some((
                entry.created_epoch.as_u64(),
                entry.deleted_epoch.map(|epoch| epoch.as_u64()),
            ));

            let lifecycle = (
                doc.node_id,
                entry.created_epoch.as_u64(),
                entry.deleted_epoch.map(|epoch| epoch.as_u64()),
            );
            doc_versions.entry(lifecycle).or_default().push(entry.len);
            if entry.deleted_epoch.is_none() {
                live_doc_count = live_doc_count
                    .checked_add(1)
                    .ok_or_else(|| "live document count overflow".to_string())?;
                live_total_length = live_total_length
                    .checked_add(u64::from(entry.len))
                    .ok_or_else(|| "live document length overflow".to_string())?;
            }
        }
    }

    let mut terms = HashSet::with_capacity(image.postings.len());
    let mut posting_token_totals: HashMap<ExactLifecycle, u64> = HashMap::new();
    for list in &image.postings {
        if list.term.is_empty() {
            return Err("text index contains an empty term".to_string());
        }
        if !terms.insert(list.term.as_str()) {
            return Err(format!(
                "text index contains duplicate term {:?}",
                list.term
            ));
        }
        if list.postings.is_empty() {
            return Err(format!("term {:?} has an empty posting list", list.term));
        }

        let mut prior_by_node: HashMap<NodeId, (u64, Option<u64>)> = HashMap::new();
        let mut lifecycle_occurrences: HashMap<ExactLifecycle, usize> = HashMap::new();
        for (position, posting) in list.postings.iter().enumerate() {
            let context = format!("term {:?} posting {position}", list.term);
            if !posting.node_id.is_valid() {
                return Err(format!("{context} uses the invalid node ID"));
            }
            if posting.term_freq == 0 {
                return Err(format!("{context} has zero term frequency"));
            }
            validate_committed_lifecycle(
                posting.created_epoch,
                posting.created_by,
                posting.deleted_epoch,
                posting.deleted_by,
                &context,
            )?;
            validate_history_order(
                prior_by_node.get(&posting.node_id).copied(),
                posting.created_epoch,
                &format!("term {:?}, node {} history", list.term, posting.node_id),
            )?;
            prior_by_node.insert(
                posting.node_id,
                (
                    posting.created_epoch.as_u64(),
                    posting.deleted_epoch.map(|epoch| epoch.as_u64()),
                ),
            );

            let lifecycle = (
                posting.node_id,
                posting.created_epoch.as_u64(),
                posting.deleted_epoch.map(|epoch| epoch.as_u64()),
            );
            let Some(doc_lens) = doc_versions.get(&lifecycle) else {
                return Err(format!(
                    "{context} has no matching document-length lifetime"
                ));
            };
            let occurrence = lifecycle_occurrences.entry(lifecycle).or_default();
            *occurrence += 1;
            if *occurrence > doc_lens.len() {
                return Err(format!(
                    "term {:?} has more postings than document versions for node {} at epoch {}",
                    list.term, posting.node_id, posting.created_epoch
                ));
            }
            if !doc_lens.iter().any(|doc_len| posting.term_freq <= *doc_len) {
                return Err(format!(
                    "{context} frequency {} exceeds every matching document length",
                    posting.term_freq
                ));
            }
            // A repeated same-epoch update can legitimately give multiple
            // historical records the same interval. Without a stable record
            // handle their per-term postings cannot be paired unambiguously;
            // preserve them in order and validate counts above. Unique
            // lifetimes (including every live lifetime) can be checked fully.
            if doc_lens.len() == 1 {
                let total = posting_token_totals.entry(lifecycle).or_default();
                *total = total
                    .checked_add(u64::from(posting.term_freq))
                    .ok_or_else(|| format!("{context} token-frequency sum overflow"))?;
            }
        }
    }

    for (lifecycle, doc_lens) in &doc_versions {
        if doc_lens.len() != 1 {
            continue;
        }
        let doc_len = doc_lens[0];
        let posting_total = posting_token_totals.get(lifecycle).copied().unwrap_or(0);
        if posting_total != u64::from(doc_len) {
            return Err(format!(
                "document {} lifetime at epoch {} has posting frequency sum {posting_total}, expected {doc_len}",
                lifecycle.0, lifecycle.1
            ));
        }
    }

    let mut aggregate_total = 0_i128;
    let mut aggregate_count = 0_i128;
    let mut positive_total = 0_i64;
    let mut negative_total = 0_i64;
    let mut positive_count = 0_i64;
    let mut negative_count = 0_i64;
    for (position, delta) in image.agg_log.iter().enumerate() {
        validate_committed_event(
            delta.epoch,
            delta.tx,
            &format!("aggregate delta {position}"),
        )?;
        if delta.d_total_len == 0 && delta.d_doc_count == 0 {
            return Err(format!("aggregate delta {position} is empty"));
        }
        // A committed delta can combine updates, inserts and deletes. Its
        // document count and token length are independent net changes: replacing
        // one long document with two short documents can increase count while
        // leaving length unchanged or decreasing it. The retained-lifetime
        // reconciliation below proves both quantities at every visible epoch.
        // Readers and GC sum subsets in stored order using i64. Bounding each
        // sign independently protects every such prefix, including cancelling
        // entries within one epoch and logs interleaved by transaction commits.
        let total_bound = if delta.d_total_len > 0 {
            &mut positive_total
        } else {
            &mut negative_total
        };
        *total_bound = total_bound.checked_add(delta.d_total_len).ok_or_else(|| {
            "aggregate total-length contributions exceed i64 accumulation bounds".to_string()
        })?;
        let count_bound = if delta.d_doc_count > 0 {
            &mut positive_count
        } else {
            &mut negative_count
        };
        *count_bound = count_bound.checked_add(delta.d_doc_count).ok_or_else(|| {
            "aggregate document-count contributions exceed i64 accumulation bounds".to_string()
        })?;
        aggregate_total = aggregate_total
            .checked_add(i128::from(delta.d_total_len))
            .ok_or_else(|| "aggregate total-length overflow".to_string())?;
        aggregate_count = aggregate_count
            .checked_add(i128::from(delta.d_doc_count))
            .ok_or_else(|| "aggregate document-count overflow".to_string())?;
    }

    if aggregate_total != i128::from(live_total_length)
        || aggregate_count != i128::from(live_doc_count)
    {
        return Err(format!(
            "aggregate log resolves to ({aggregate_total} tokens, {aggregate_count} documents), expected ({live_total_length} tokens, {live_doc_count} documents)"
        ));
    }

    if let Some(epoch) = aggregate_history_mismatch(image, image.retained_from.as_u64())? {
        return Err(format!(
            "aggregate history differs from retained document lifetimes at epoch {epoch}, retained from {}",
            image.retained_from,
        ));
    }
    Ok(())
}

/// Returns the first epoch where aggregate and document transitions differ.
/// Grouping all events at an epoch preserves updates/deletes within one commit:
/// intermediate same-epoch states are never visible to a committed reader.
fn aggregate_history_mismatch(
    image: &ExactTextIndexImage,
    gc_cutoff: u64,
) -> std::result::Result<Option<u64>, String> {
    let event_count = image
        .doc_lengths
        .iter()
        .try_fold(image.agg_log.len(), |count, doc| {
            doc.history.iter().try_fold(count, |count, entry| {
                count
                    .checked_add(1 + usize::from(entry.deleted_epoch.is_some()))
                    .ok_or_else(|| "aggregate history event count overflow".to_string())
            })
        })?;
    let mut events = Vec::new();
    events
        .try_reserve_exact(event_count)
        .map_err(|error| format!("cannot reserve aggregate history validation events: {error}"))?;
    for entry in image.doc_lengths.iter().flat_map(|doc| &doc.history) {
        let created = entry.created_epoch.as_u64();
        events.push((
            if created <= gc_cutoff { 0 } else { created },
            i128::from(entry.len),
            1_i128,
        ));
        if let Some(deleted) = entry.deleted_epoch {
            events.push((deleted.as_u64(), -i128::from(entry.len), -1));
        }
    }
    for delta in &image.agg_log {
        events.push((
            if delta.epoch.as_u64() <= gc_cutoff {
                0
            } else {
                delta.epoch.as_u64()
            },
            -i128::from(delta.d_total_len),
            -i128::from(delta.d_doc_count),
        ));
    }
    events.sort_unstable_by_key(|event| event.0);

    let mut epoch = 0;
    let mut total_difference = 0_i128;
    let mut count_difference = 0_i128;
    for (next_epoch, total, count) in events {
        if next_epoch != epoch {
            if total_difference != 0 || count_difference != 0 {
                return Ok(Some(epoch));
            }
            epoch = next_epoch;
        }
        total_difference = total_difference
            .checked_add(total)
            .ok_or_else(|| "aggregate history length comparison overflow".to_string())?;
        count_difference = count_difference
            .checked_add(count)
            .ok_or_else(|| "aggregate history count comparison overflow".to_string())?;
    }
    Ok((total_difference != 0 || count_difference != 0).then_some(epoch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_floor_checks_empty_queries_and_survives_exact_images()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        let node = NodeId::new(1);
        index.insert_versioned(node, "alpha beta", EpochId::new(1), None);
        index.insert_versioned(NodeId::new(2), "gamma delta", EpochId::new(1), None);
        index.insert_versioned(node, "alpha epsilon", EpochId::new(5), None);
        index.remove_versioned(NodeId::new(2), EpochId::new(7), None);
        let mut expected = Vec::new();
        for epoch in [3, 5, 7, 9].map(EpochId::new) {
            expected.push((
                epoch,
                index.total_length_at(epoch, TransactionId::INVALID)?,
                index.doc_count_at(epoch, TransactionId::INVALID)?,
                index.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                index
                    .score_document_visible(
                        node,
                        "alpha",
                        epoch,
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .ok_or("expected indexed document")?
                    .to_bits(),
            ));
        }
        assert_eq!(expected[0].4, std::f64::consts::LN_2.to_bits());
        let latest = index.search("alpha", 10);
        index.gc(EpochId::new(3))?;
        assert_eq!(index.retained_from(), EpochId::new(3));
        assert_eq!(index.search("alpha", 10), latest);
        let image = index.exact_committed_image()?;
        let bytes = bincode::serde::encode_to_vec(&image, bincode::config::standard())?;
        let mut restored = InvertedIndex::new(BM25Config::default());
        let prepared = InvertedIndex::prepare_exact_image(image.clone())?;
        let proof = restored
            .pin_prepared_restore()
            .ok_or("restore proof missing")?;
        restored.install_prepared_image(prepared, &proof);
        drop(proof);
        let empty = FxHashSet::default();
        for candidate in [&index, &restored] {
            assert_eq!(candidate.retained_from(), EpochId::new(3));
            for &(epoch, length, count, average, score) in &expected {
                assert_eq!(
                    candidate.total_length_at(epoch, TransactionId::INVALID)?,
                    length
                );
                assert_eq!(
                    candidate.doc_count_at(epoch, TransactionId::INVALID)?,
                    count
                );
                assert_eq!(
                    candidate.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                    average
                );
                assert_eq!(
                    candidate
                        .score_document_visible(
                            node,
                            "alpha",
                            epoch,
                            TransactionId::INVALID,
                            None,
                            false
                        )?
                        .ok_or("retained score disappeared")?
                        .to_bits(),
                    score
                );
                assert_eq!(
                    candidate.search_visible(
                        "alpha",
                        10,
                        epoch,
                        TransactionId::INVALID,
                        &[],
                        &empty
                    )?[0]
                        .1
                        .to_bits(),
                    score
                );
                assert_eq!(
                    candidate.search_with_threshold_visible(
                        "alpha",
                        0.0,
                        epoch,
                        TransactionId::INVALID,
                        &[],
                        &empty
                    )?[0]
                        .1
                        .to_bits(),
                    score
                );
            }
            let old = EpochId::new(2);
            assert!(matches!(
                candidate.total_length_at(old, TransactionId::INVALID),
                Err(Error::InvalidValue(_))
            ));
            assert!(matches!(
                candidate.doc_count_at(old, TransactionId::INVALID),
                Err(Error::InvalidValue(_))
            ));
            assert!(matches!(
                candidate.avgdl_at(old, TransactionId::INVALID),
                Err(Error::InvalidValue(_))
            ));
            assert!(matches!(
                candidate.search_visible("", 0, old, TransactionId::INVALID, &[], &empty),
                Err(Error::InvalidValue(_))
            ));
            assert!(matches!(
                candidate.search_with_threshold_visible(
                    "",
                    f64::INFINITY,
                    old,
                    TransactionId::INVALID,
                    &[],
                    &empty
                ),
                Err(Error::InvalidValue(_))
            ));
            assert!(matches!(
                candidate.score_document_visible(
                    NodeId::new(99),
                    "",
                    old,
                    TransactionId::INVALID,
                    None,
                    true
                ),
                Err(Error::InvalidValue(_))
            ));
            assert_eq!(
                bincode::serde::encode_to_vec(
                    candidate.exact_committed_image()?,
                    bincode::config::standard()
                )?,
                bytes
            );
        }
        #[cfg(feature = "lpg")]
        {
            let fork = index.exact_runtime_fork();
            assert_eq!(fork.retained_from(), EpochId::new(3));
            assert_eq!(
                bincode::serde::encode_to_vec(
                    fork.exact_committed_image()?,
                    bincode::config::standard()
                )?,
                bytes
            );
        }
        for floor in [EpochId::INITIAL, EpochId::PENDING, EpochId::new(6)] {
            let mut malformed = image.clone();
            malformed.retained_from = floor;
            assert!(InvertedIndex::prepare_exact_image(malformed).is_err());
        }
        assert!(matches!(
            index.gc(EpochId::PENDING),
            Err(Error::InvalidValue(_))
        ));
        index.gc(EpochId::new(1))?;
        assert_eq!(index.retained_from(), EpochId::new(3));
        assert_eq!(
            bincode::serde::encode_to_vec(
                index.exact_committed_image()?,
                bincode::config::standard()
            )?,
            bytes
        );
        index.gc(EpochId::new(5))?;
        assert_eq!(index.retained_from(), EpochId::new(5));
        assert!(
            index
                .doc_count_at(EpochId::new(3), TransactionId::INVALID)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn gc_preserves_pending_records_and_empty_index_admission()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        let deleted = NodeId::new(1);
        let pending = NodeId::new(2);
        let deleter = TransactionId::new(11);
        let creator = TransactionId::new(12);
        index.insert_versioned(deleted, "alpha", EpochId::new(1), None);
        index.remove_versioned(deleted, EpochId::PENDING, Some(deleter));
        index.insert_versioned(pending, "gamma", EpochId::PENDING, Some(creator));
        let postings = format!(
            "{:?}|{:?}",
            index.postings["alpha"], index.postings["gamma"]
        );
        let lengths = format!(
            "{:?}|{:?}",
            index.doc_lengths[&deleted], index.doc_lengths[&pending]
        );
        let deltas = index
            .agg_log
            .iter()
            .filter(|delta| delta.tx.is_some())
            .map(|delta| format!("{delta:?}"))
            .collect::<Vec<_>>();
        index.gc(EpochId::new(3))?;
        assert_eq!(
            format!(
                "{:?}|{:?}",
                index.postings["alpha"], index.postings["gamma"]
            ),
            postings
        );
        assert_eq!(
            format!(
                "{:?}|{:?}",
                index.doc_lengths[&deleted], index.doc_lengths[&pending]
            ),
            lengths
        );
        assert_eq!(
            index
                .agg_log
                .iter()
                .filter(|delta| delta.tx.is_some())
                .map(|delta| format!("{delta:?}"))
                .collect::<Vec<_>>(),
            deltas
        );
        assert_eq!(index.doc_count_at(EpochId::new(3), deleter)?, 0);
        assert_eq!(index.doc_count_at(EpochId::new(3), creator)?, 2);
        assert_eq!(
            index.doc_count_at(EpochId::new(3), TransactionId::INVALID)?,
            1
        );
        let mut empty = InvertedIndex::new(BM25Config::default());
        empty.gc(EpochId::new(3))?;
        assert!(
            empty
                .search_visible(
                    "",
                    0,
                    EpochId::new(2),
                    TransactionId::INVALID,
                    &[],
                    &FxHashSet::default()
                )
                .is_err()
        );
        assert_eq!(
            empty.total_length_at(EpochId::new(3), TransactionId::INVALID)?,
            0
        );
        assert!(empty.search("", 0).is_empty());
        assert!(InvertedIndex::prepare_exact_image(empty.exact_committed_image()?).is_ok());
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn gc_denied_authority_preserves_floor_and_image()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::graph::write_permit::{WriteAuthority, with_authority};
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert_versioned(NodeId::new(1), "alpha", EpochId::new(1), None);
        let owner = WriteAuthority::new();
        let transition = InvertedIndex::pin_scope_transition();
        assert!(index.bind_under_transition(41, 7, &transition));
        assert!(index.seal_with_scope_under_transition(owner.scope().get(), &transition));
        drop(transition);
        let before = bincode::serde::encode_to_vec(
            index.exact_committed_image()?,
            bincode::config::standard(),
        )?;
        assert!(matches!(
            index.gc(EpochId::new(3)),
            Err(Error::Transaction(TransactionError::ReadOnly))
        ));
        assert_eq!(index.retained_from(), EpochId::INITIAL);
        assert_eq!(
            bincode::serde::encode_to_vec(
                index.exact_committed_image()?,
                bincode::config::standard()
            )?,
            before
        );
        with_authority(&owner, || index.gc(EpochId::new(3)))?;
        assert_eq!(index.retained_from(), EpochId::new(3));
        Ok(())
    }

    #[test]
    fn exact_image_rejects_fabricated_aggregate_histories() {
        let mut rejected = Vec::new();
        for deltas in [vec![1, -1], vec![i64::MAX, i64::MAX, -i64::MAX, -i64::MAX]] {
            let index = InvertedIndex::new(BM25Config::default());
            let mut image = index.exact_committed_image().unwrap();
            image.agg_log = deltas
                .into_iter()
                .enumerate()
                .map(|(position, d_total_len)| ExactAggDelta {
                    epoch: EpochId::new(position as u64 + 1),
                    tx: None,
                    d_total_len,
                    d_doc_count: 0,
                })
                .collect();

            rejected.push(InvertedIndex::prepare_exact_image(image).is_err());
        }
        assert_eq!(
            rejected,
            [true, true],
            "both fabricated histories must fail"
        );
    }

    #[test]
    fn exact_image_rejects_same_epoch_aggregate_accumulation_overflow() {
        let mut image = InvertedIndex::new(BM25Config::default())
            .exact_committed_image()
            .unwrap();
        image.agg_log = [i64::MAX, i64::MAX, -i64::MAX, -i64::MAX]
            .into_iter()
            .map(|d_total_len| ExactAggDelta {
                epoch: EpochId::new(1),
                tx: None,
                d_total_len,
                d_doc_count: 0,
            })
            .collect();

        assert!(
            InvertedIndex::prepare_exact_image(image).is_err(),
            "same-epoch cancellation must not admit an overflowing reader prefix"
        );
    }

    fn coalesced_commit_fixture(
        before: &[usize],
        after: &[usize],
    ) -> std::result::Result<(InvertedIndex, ExactTextIndexImage), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::with_simple_tokenizer(BM25Config::default(), 7);
        for (position, &length) in before.iter().enumerate() {
            index.insert_versioned(
                NodeId::new(u64::try_from(position)?),
                &"retained ".repeat(length),
                EpochId::new(1),
                None,
            );
        }
        for (position, &length) in after.iter().enumerate() {
            index.insert_versioned(
                NodeId::new(u64::try_from(position)?),
                &"retained ".repeat(length),
                EpochId::new(5),
                None,
            );
        }
        for position in after.len()..before.len() {
            assert!(index.remove_versioned(
                NodeId::new(u64::try_from(position)?),
                EpochId::new(5),
                None,
            ));
        }
        index.gc(EpochId::new(3))?;
        let mut image = index.exact_committed_image()?;
        let d_total_len = image
            .agg_log
            .iter()
            .filter(|delta| delta.epoch == EpochId::new(5))
            .map(|delta| delta.d_total_len)
            .sum();
        let d_doc_count = image
            .agg_log
            .iter()
            .filter(|delta| delta.epoch == EpochId::new(5))
            .map(|delta| delta.d_doc_count)
            .sum();
        image.agg_log.retain(|delta| delta.epoch != EpochId::new(5));
        image.agg_log.push(ExactAggDelta {
            epoch: EpochId::new(5),
            tx: None,
            d_total_len,
            d_doc_count,
        });
        Ok((index, image))
    }

    #[test]
    fn exact_image_accepts_coalesced_zero_and_opposite_sign_commit_deltas()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for (before, after, expected_delta) in [
            (&[2][..], &[1, 1][..], (0, 1)),
            (&[5][..], &[1, 1][..], (-3, 1)),
            (&[1, 1][..], &[2][..], (0, -1)),
            (&[1, 1][..], &[5][..], (3, -1)),
        ] {
            let (mut source, image) = coalesced_commit_fixture(before, after)?;
            let commit = image.agg_log.last().ok_or("missing coalesced commit")?;
            assert_eq!((commit.d_total_len, commit.d_doc_count), expected_delta);
            let bytes = bincode::serde::encode_to_vec(&image, bincode::config::standard())?;
            let prepared = InvertedIndex::prepare_exact_image(image)?;
            let mut restored = InvertedIndex::new(BM25Config::default());
            let proof = restored
                .pin_prepared_restore()
                .ok_or("restore proof missing")?;
            restored.install_prepared_image(prepared, &proof);
            drop(proof);
            assert_eq!(restored.retained_from(), EpochId::new(3));
            assert_eq!(
                bincode::serde::encode_to_vec(
                    restored.exact_committed_image()?,
                    bincode::config::standard()
                )?,
                bytes
            );
            for epoch in [3, 4, 5, 6].map(EpochId::new) {
                assert_eq!(
                    restored.total_length_at(epoch, TransactionId::INVALID)?,
                    source.total_length_at(epoch, TransactionId::INVALID)?
                );
                assert_eq!(
                    restored.doc_count_at(epoch, TransactionId::INVALID)?,
                    source.doc_count_at(epoch, TransactionId::INVALID)?
                );
                assert_eq!(
                    restored.avgdl_at(epoch, TransactionId::INVALID)?.to_bits(),
                    source.avgdl_at(epoch, TransactionId::INVALID)?.to_bits()
                );
                for position in 0..before.len().max(after.len()) {
                    let node = NodeId::new(u64::try_from(position)?);
                    let score = |index: &InvertedIndex| {
                        index
                            .score_document_visible(
                                node,
                                "retained",
                                epoch,
                                TransactionId::INVALID,
                                None,
                                false,
                            )
                            .map(|score| score.map(f64::to_bits))
                    };
                    assert_eq!(score(&restored)?, score(&source)?);
                }
            }
            source.gc(EpochId::new(5))?;
            restored.gc(EpochId::new(5))?;
            assert_eq!(
                bincode::serde::encode_to_vec(
                    restored.exact_committed_image()?,
                    bincode::config::standard()
                )?,
                bincode::serde::encode_to_vec(
                    source.exact_committed_image()?,
                    bincode::config::standard()
                )?
            );
        }
        Ok(())
    }

    #[test]
    fn exact_image_coalesced_deltas_still_require_exact_retained_lifetimes()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for (before, after) in [
            (&[2][..], &[1, 1][..]),
            (&[5][..], &[1, 1][..]),
            (&[1, 1][..], &[5][..]),
        ] {
            let (_, image) = coalesced_commit_fixture(before, after)?;
            assert!(InvertedIndex::prepare_exact_image(image.clone()).is_ok());
            // Final sums remain correct, but neither moving the transition
            // later nor hiding it in the retained baseline is permissible.
            for shifted in [EpochId::INITIAL, EpochId::new(6)] {
                let mut malformed = image.clone();
                malformed.agg_log.last_mut().ok_or("missing commit")?.epoch = shifted;
                let error = validate_exact_image(&malformed)
                    .expect_err("shifted retained transition must fail");
                assert!(
                    error.contains("aggregate history differs from retained document lifetimes"),
                    "{error}"
                );
            }
        }
        // Cancelling forged zero/opposite-sign events preserve final totals
        // too, but have no corresponding retained document transitions.
        for (length, count) in [(0, 1), (-3, 1), (3, -1)] {
            let mut image = InvertedIndex::new(BM25Config::default()).exact_committed_image()?;
            image.retained_from = EpochId::new(3);
            image.agg_log = vec![
                ExactAggDelta {
                    epoch: EpochId::new(5),
                    tx: None,
                    d_total_len: length,
                    d_doc_count: count,
                },
                ExactAggDelta {
                    epoch: EpochId::new(6),
                    tx: None,
                    d_total_len: -length,
                    d_doc_count: -count,
                },
            ];
            let error = validate_exact_image(&image)
                .expect_err("fabricated retained transitions must fail");
            assert!(
                error.contains("aggregate history differs from retained document lifetimes"),
                "{error}"
            );
        }
        Ok(())
    }

    #[test]
    fn exact_image_rejects_shifted_retained_transitions_after_gc()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert_versioned(NodeId::new(1), "alpha beta", EpochId::new(1), None);
        index.insert_versioned(NodeId::new(1), "alpha beta gamma", EpochId::new(5), None);
        index.remove_versioned(NodeId::new(1), EpochId::new(7), None);
        index.gc(EpochId::new(3))?;
        let mut image = index.exact_committed_image().unwrap();
        for delta in &mut image.agg_log {
            if delta.epoch == EpochId::new(5) {
                delta.epoch = EpochId::new(6);
            }
        }
        assert!(
            InvertedIndex::prepare_exact_image(image).is_err(),
            "a surviving update cannot be hidden inside an inferred GC baseline"
        );
        Ok(())
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(
            NodeId::new(1),
            "the quick brown fox jumps over the lazy dog",
        );
        index.insert(NodeId::new(2), "a fast red car drives on the highway");
        index.insert(NodeId::new(3), "the brown dog sleeps all day");

        let results = index.search("brown dog", 10);
        assert!(!results.is_empty());
        // Node 3 mentions both "brown" and "dog" in a shorter document
        assert_eq!(results[0].0, NodeId::new(3));
    }

    #[test]
    fn test_empty_index_search() {
        let index = InvertedIndex::new(BM25Config::default());
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_query() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_stop_word_only_query() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("the a an", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        index.insert(NodeId::new(2), "hello rust");

        assert_eq!(index.len(), 2);
        assert!(index.remove(NodeId::new(1)));
        assert_eq!(index.len(), 1);

        let results = index.search("hello", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(2));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut index = InvertedIndex::new(BM25Config::default());
        assert!(!index.remove(NodeId::new(999)));
    }

    #[test]
    fn test_reinsert() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "old text");
        index.insert(NodeId::new(1), "new text completely different");

        assert_eq!(index.len(), 1);
        let results = index.search("old", 10);
        assert!(results.is_empty());

        let results = index.search("completely different", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_contains() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");

        assert!(index.contains(NodeId::new(1)));
        assert!(!index.contains(NodeId::new(2)));
    }

    #[test]
    fn test_term_count() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        index.insert(NodeId::new(2), "hello rust");

        // "hello", "world", "rust" (stop words removed)
        assert_eq!(index.term_count(), 3);
    }

    #[test]
    fn test_k_limit() {
        let mut index = InvertedIndex::new(BM25Config::default());
        for i in 1..=10 {
            index.insert(NodeId::new(i), &format!("document number {}", i));
        }

        let results = index.search("document", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_bm25_scoring_prefers_shorter_docs() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // Short doc with the term
        index.insert(NodeId::new(1), "rust database");
        // Long doc with the same term buried in noise
        index.insert(
            NodeId::new(2),
            "rust programming language systems web server framework database engine query optimizer",
        );

        let results = index.search("rust database", 10);
        assert_eq!(results.len(), 2);
        // Shorter doc should score higher (length normalization)
        assert_eq!(results[0].0, NodeId::new(1));
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_no_match() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("nonexistent term", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_idf_weighting() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // "common" appears in all docs, "rare" only in one
        index.insert(NodeId::new(1), "common rare word");
        index.insert(NodeId::new(2), "common another word");
        index.insert(NodeId::new(3), "common third word");

        let results = index.search("rare", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));

        // "common" matches all three
        let results = index.search("common", 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_score_document_matches_search() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(
            NodeId::new(1),
            "the quick brown fox jumps over the lazy dog",
        );
        index.insert(NodeId::new(2), "a fast red car drives on the highway");
        index.insert(NodeId::new(3), "the brown dog sleeps all day");

        let query = "brown dog";
        let search_results = index.search(query, 10);

        // Verify score_document returns the same score as search() for matching docs
        for (node_id, search_score) in &search_results {
            let doc_score = index.score_document(*node_id, query);
            assert!(
                (doc_score - search_score).abs() < 1e-10,
                "score_document({:?}) = {doc_score} but search gave {search_score}",
                node_id
            );
        }

        // Node 2 has no matching terms — should score 0.0
        let no_match_score = index.score_document(NodeId::new(2), query);
        assert_eq!(no_match_score, 0.0, "non-matching doc should score 0.0");

        // Non-existent doc should score 0.0
        let nonexistent_score = index.score_document(NodeId::new(999), query);
        assert_eq!(nonexistent_score, 0.0, "non-existent doc should score 0.0");
    }

    #[test]
    fn visible_zero_score_is_distinct_from_absent_document()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        let node = NodeId::new(1);
        let epoch = EpochId::new(1);
        let tx = TransactionId::INVALID;
        index.insert_versioned(node, "other stuff", epoch, None);
        index.insert_versioned(NodeId::new(2), "rust guide", epoch, None);

        // Cover both a term present elsewhere in the corpus and an unknown term.
        for query in ["rust", "unknown"] {
            assert_eq!(
                index.score_document_visible(node, query, epoch, tx, None, false)?,
                Some(index.score_document(node, query))
            );
            assert_eq!(index.score_document(node, query), 0.0);
            assert_eq!(
                index.score_document_visible(NodeId::new(99), query, epoch, tx, None, false)?,
                None
            );
            assert_eq!(
                index.score_document_visible(node, query, EpochId::new(0), tx, None, false)?,
                None
            );
            assert_eq!(
                index.score_document_visible(node, query, epoch, tx, None, true)?,
                None
            );
        }
        index.remove_versioned(node, EpochId::new(2), None);
        assert_eq!(
            index.score_document_visible(node, "rust", EpochId::new(2), tx, None, false)?,
            None
        );
        assert_eq!(
            index.score_document_visible(node, "rust", epoch, tx, None, false)?,
            Some(0.0)
        );
        index.gc(epoch)?;
        assert!(
            index
                .score_document_visible(node, "unknown", EpochId::new(0), tx, None, false)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_search_with_threshold() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "rust graph database query engine");
        index.insert(NodeId::new(2), "python web framework django flask");
        index.insert(NodeId::new(3), "rust systems programming language");
        index.insert(NodeId::new(4), "graph theory algorithms data structures");
        index.insert(
            NodeId::new(5),
            "database indexing storage engine optimization",
        );

        let query = "rust graph database";

        // Get search results to calibrate the threshold
        let search_results = index.search(query, 10);
        assert!(
            search_results.len() >= 2,
            "need at least 2 matching docs for this test"
        );

        // Use the score of the second-highest result as our mid threshold
        let mid_threshold = search_results[1].1;

        // threshold=0 should return all matching docs (same set as search with no k limit)
        let all_results = index.search_with_threshold(query, 0.0);
        assert_eq!(
            all_results.len(),
            search_results.len(),
            "threshold=0 should return all matching docs"
        );

        // Results should be sorted descending by score
        for i in 1..all_results.len() {
            assert!(
                all_results[i - 1].1 >= all_results[i].1,
                "results should be sorted descending"
            );
        }

        // mid_threshold should filter out lower-scoring docs
        let filtered = index.search_with_threshold(query, mid_threshold);
        assert!(
            filtered.len() <= search_results.len(),
            "mid-threshold should not exceed total matches"
        );
        for (_, score) in &filtered {
            assert!(
                *score >= mid_threshold,
                "all returned docs should score >= threshold"
            );
        }

        // Very high threshold should return nothing
        let empty_results = index.search_with_threshold(query, 1_000_000.0);
        assert!(
            empty_results.is_empty(),
            "very high threshold should return no results"
        );

        // Empty query should return nothing
        let empty_query_results = index.search_with_threshold("", 0.0);
        assert!(
            empty_query_results.is_empty(),
            "empty query should return no results"
        );
    }

    // ── As-of-epoch aggregate tests (Task 2 TDD) ──────────────────────────

    /// Insert two docs at different epochs; verify `doc_count_at` and
    /// `total_length_at` reflect only the docs visible at each epoch.
    #[test]
    fn test_doc_count_at_epoch_boundary() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        // 3 tokens: "hello", "world", "one"  → len 3
        index.insert_versioned(NodeId::new(1), "hello world one", EpochId::new(1), None);
        // 2 tokens: "foo", "bar"  → len 2
        index.insert_versioned(NodeId::new(2), "foo bar", EpochId::new(2), None);

        // At epoch 1 only node 1 is visible.
        assert_eq!(
            index.doc_count_at(EpochId::new(1), TransactionId::INVALID)?,
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(1), TransactionId::INVALID)?,
            3
        );

        // At epoch 2 both nodes are visible.
        assert_eq!(
            index.doc_count_at(EpochId::new(2), TransactionId::INVALID)?,
            2
        );
        assert_eq!(
            index.total_length_at(EpochId::new(2), TransactionId::INVALID)?,
            5
        );
        Ok(())
    }

    /// A doc deleted after E1 must still be counted at E1.
    #[test]
    fn test_doc_still_visible_at_epoch_before_deletion()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        // "alpha beta gamma" → 3 tokens
        index.insert_versioned(NodeId::new(1), "alpha beta gamma", EpochId::new(1), None);
        // Delete at epoch 5.
        index.remove_versioned(NodeId::new(1), EpochId::new(5), None);

        // At epoch 3 (before deletion) it is still visible.
        assert_eq!(
            index.doc_count_at(EpochId::new(3), TransactionId::INVALID)?,
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(3), TransactionId::INVALID)?,
            3
        );

        // At epoch 5 (at deletion) it is gone.
        assert_eq!(
            index.doc_count_at(EpochId::new(5), TransactionId::INVALID)?,
            0
        );
        assert_eq!(
            index.total_length_at(EpochId::new(5), TransactionId::INVALID)?,
            0
        );
        Ok(())
    }

    /// `avgdl_at` uses the as-of-E totals.
    #[test]
    fn test_avgdl_at_epoch() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        // "hello world one" → 3 tokens (len 3)
        index.insert_versioned(NodeId::new(1), "hello world one", EpochId::new(1), None);
        // "foo bar baz qux" → 4 tokens (len 4)
        index.insert_versioned(NodeId::new(2), "foo bar baz qux", EpochId::new(3), None);

        // At epoch 1: only doc1, avgdl = 3/1 = 3.0
        let avgdl_e1 = index.avgdl_at(EpochId::new(1), TransactionId::INVALID)?;
        assert!(
            (avgdl_e1 - 3.0).abs() < 1e-10,
            "avgdl@1 expected 3.0, got {avgdl_e1}"
        );

        // At epoch 3: both docs, avgdl = (3+4)/2 = 3.5
        let avgdl_e3 = index.avgdl_at(EpochId::new(3), TransactionId::INVALID)?;
        assert!(
            (avgdl_e3 - 3.5).abs() < 1e-10,
            "avgdl@3 expected 3.5, got {avgdl_e3}"
        );
        Ok(())
    }

    /// A doc whose length changes between epochs must report the as-of-E length.
    #[test]
    fn test_avgdl_changes_after_doc_update() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let mut index = InvertedIndex::new(BM25Config::default());
        // Insert node 1 at epoch 1 with 2 tokens.
        index.insert_versioned(NodeId::new(1), "alpha beta", EpochId::new(1), None);
        // Re-insert (update) node 1 at epoch 5 with 4 tokens.
        index.insert_versioned(
            NodeId::new(1),
            "alpha beta gamma delta",
            EpochId::new(5),
            None,
        );

        // At epoch 1: len=2 (original), count=1, avgdl=2.0
        assert_eq!(
            index.doc_count_at(EpochId::new(1), TransactionId::INVALID)?,
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(1), TransactionId::INVALID)?,
            2
        );
        let avgdl_e1 = index.avgdl_at(EpochId::new(1), TransactionId::INVALID)?;
        assert!(
            (avgdl_e1 - 2.0).abs() < 1e-10,
            "avgdl@1 expected 2.0, got {avgdl_e1}"
        );

        // At epoch 5: len=4 (updated), count=1, avgdl=4.0
        assert_eq!(
            index.doc_count_at(EpochId::new(5), TransactionId::INVALID)?,
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(5), TransactionId::INVALID)?,
            4
        );
        let avgdl_e5 = index.avgdl_at(EpochId::new(5), TransactionId::INVALID)?;
        assert!(
            (avgdl_e5 - 4.0).abs() < 1e-10,
            "avgdl@5 expected 4.0, got {avgdl_e5}"
        );
        Ok(())
    }

    /// Committed-latest `avgdl_at(COMMITTED_EPOCH, INVALID)` == `total_length/n`
    /// for epoch-0 legacy inserts — i.e., behaviour-preserving.
    #[test]
    #[allow(clippy::cast_possible_truncation)] // doc_count is 2 in this test
    fn test_avgdl_at_committed_epoch_matches_legacy()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world"); // 2 tokens
        index.insert(NodeId::new(2), "foo bar baz"); // 3 tokens

        let avgdl = index.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID)?;
        // (2+3)/2 = 2.5
        assert!(
            (avgdl - 2.5).abs() < 1e-10,
            "avgdl at committed epoch expected 2.5, got {avgdl}"
        );
        // doc_count matches len()
        assert_eq!(
            index.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID)? as usize,
            index.len()
        );
        Ok(())
    }

    /// Own-tx pending inserts are visible only to the inserting tx.
    #[test]
    fn test_doc_count_at_own_tx_pending() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        let tx7 = TransactionId::new(7);
        // Pending insert by tx 7.
        index.insert_versioned(NodeId::new(1), "hello world", EpochId::PENDING, Some(tx7));

        // tx 7 can see its own pending insert.
        assert_eq!(index.doc_count_at(EpochId::new(10), tx7)?, 1);
        // tx 8 cannot.
        assert_eq!(
            index.doc_count_at(EpochId::new(10), TransactionId::new(8))?,
            0
        );
        // Committed-latest reader cannot.
        assert_eq!(
            index.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID)?,
            0
        );
        Ok(())
    }

    /// `avgdl_at` returns 0.0 when the corpus is empty.
    #[test]
    fn test_avgdl_at_empty_index() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let index = InvertedIndex::new(BM25Config::default());
        assert_eq!(
            index.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID)?,
            0.0
        );
        Ok(())
    }

    // ── GC tests (Task 6 TDD) ──────────────────────────────────────────────

    /// `gc(horizon)` drops postings for a doc that was fully deleted below the
    /// horizon, but keeps postings for docs deleted ABOVE the horizon (a snapshot
    /// at horizon–1 still needs them) and for live (never-deleted) docs.
    ///
    /// The aggregate log after `gc` still yields the correct `avgdl_at(E)` for
    /// any `E >= horizon`.
    #[test]
    fn gc_drops_postings_deleted_below_horizon()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        // Doc 1 inserted at E1=1, removed at E2=3. horizon=E3=10 (>= E2).
        let e1 = EpochId::new(1);
        let e2 = EpochId::new(3);
        let e3 = EpochId::new(10);
        // Doc 2 inserted at E1=1, removed at E4=20 (above the horizon E3=10).
        let e4 = EpochId::new(20);
        // Doc 3 inserted at E1=1, never deleted — always live.

        index.insert_versioned(NodeId::new(1), "hello world", e1, None);
        index.insert_versioned(NodeId::new(2), "foo bar baz", e1, None);
        index.insert_versioned(NodeId::new(3), "live forever doc", e1, None);

        index.remove_versioned(NodeId::new(1), e2, None); // deleted below horizon
        index.remove_versioned(NodeId::new(2), e4, None); // deleted ABOVE horizon

        // Before gc: all posting lists exist.
        assert!(
            index
                .postings
                .values()
                .any(|pl| pl.postings.iter().any(|p| p.node_id == NodeId::new(1)))
        );

        // gc with horizon = e3 (>= e2, < e4).
        index.gc(e3)?;

        // Doc 1 postings MUST be gone (deleted_epoch=3 <= horizon=10).
        for pl in index.postings.values() {
            for p in &pl.postings {
                assert_ne!(
                    p.node_id,
                    NodeId::new(1),
                    "doc 1 posting must be removed after gc below horizon"
                );
            }
        }

        // Doc 2 postings MUST still be present (deleted_epoch=20 > horizon=10).
        let doc2_present = index
            .postings
            .values()
            .any(|pl| pl.postings.iter().any(|p| p.node_id == NodeId::new(2)));
        assert!(
            doc2_present,
            "doc 2 posting must be retained — still snapshot-visible below horizon"
        );

        // Doc 3 postings MUST still be present (live, no deleted_epoch).
        let doc3_present = index
            .postings
            .values()
            .any(|pl| pl.postings.iter().any(|p| p.node_id == NodeId::new(3)));
        assert!(doc3_present, "live doc 3 must never be dropped by gc");

        // Doc 1 doc_lengths entry MUST be gone.
        assert!(
            index
                .doc_lengths
                .get(&NodeId::new(1))
                .map_or(true, |v| v.is_empty()),
            "doc 1 doc_lengths entry must be removed after gc"
        );

        // Aggregate at horizon and above must still be correct.
        // At e3=10: doc1 deleted at e2=3 (not visible), doc2 deleted at e4=20 (visible),
        // doc3 live → 2 docs, total_len = "foo bar baz"(3) + "live forever doc"(3) = 6.
        let count_at_e3 = index.doc_count_at(e3, TransactionId::INVALID)?;
        assert_eq!(count_at_e3, 2, "doc_count_at(e3) must be 2 after gc");
        let total_at_e3 = index.total_length_at(e3, TransactionId::INVALID)?;
        assert_eq!(total_at_e3, 6, "total_length_at(e3) must be 6 after gc");
        Ok(())
    }

    /// After `gc(horizon)` with `horizon < deletion_epoch`, the posting is KEPT:
    /// a snapshot at any epoch between insert and delete still needs to see it.
    #[test]
    fn gc_keeps_posting_deleted_above_horizon()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        let e1 = EpochId::new(1);
        let e5 = EpochId::new(5);
        let e2 = EpochId::new(2); // horizon below deletion

        index.insert_versioned(NodeId::new(99), "keep me please", e1, None);
        index.remove_versioned(NodeId::new(99), e5, None); // deleted at 5

        index.gc(e2)?; // horizon = 2, below deletion epoch 5

        // Posting must still be present.
        let present = index
            .postings
            .values()
            .any(|pl| pl.postings.iter().any(|p| p.node_id == NodeId::new(99)));
        assert!(present, "posting deleted above horizon must be retained");
        Ok(())
    }

    /// `gc(horizon)` compacts the aggregate log: after compaction,
    /// `doc_count_at(E)` and `total_length_at(E)` for `E >= horizon` equal the
    /// pre-gc values; reads below the retained horizon are rejected.
    #[test]
    fn gc_compacts_aggregate_log() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());

        // Insert/remove several docs across multiple epochs.
        for i in 1u64..=5 {
            index.insert_versioned(
                NodeId::new(i),
                &format!("document {i} with some words"),
                EpochId::new(i),
                None,
            );
        }
        // Remove doc 1 and doc 2 at epoch 6.
        index.remove_versioned(NodeId::new(1), EpochId::new(6), None);
        index.remove_versioned(NodeId::new(2), EpochId::new(6), None);

        let horizon = EpochId::new(6);

        // Capture pre-gc aggregate values at and above horizon.
        let pre_count_at_h = index.doc_count_at(horizon, TransactionId::INVALID)?;
        let pre_total_at_h = index.total_length_at(horizon, TransactionId::INVALID)?;
        let pre_count_at_100 = index.doc_count_at(EpochId::new(100), TransactionId::INVALID)?;
        let pre_total_at_100 = index.total_length_at(EpochId::new(100), TransactionId::INVALID)?;

        let pre_agg_len = index.agg_log.len();

        index.gc(horizon)?;

        let post_agg_len = index.agg_log.len();
        assert!(
            post_agg_len < pre_agg_len,
            "gc must compact the aggregate log (pre={pre_agg_len}, post={post_agg_len})"
        );

        // Aggregate values at and above horizon must be preserved.
        assert_eq!(
            index.doc_count_at(horizon, TransactionId::INVALID)?,
            pre_count_at_h,
            "doc_count_at(horizon) must be preserved"
        );
        assert_eq!(
            index.total_length_at(horizon, TransactionId::INVALID)?,
            pre_total_at_h,
            "total_length_at(horizon) must be preserved"
        );
        assert_eq!(
            index.doc_count_at(EpochId::new(100), TransactionId::INVALID)?,
            pre_count_at_100,
            "doc_count_at(100) must be preserved"
        );
        assert_eq!(
            index.total_length_at(EpochId::new(100), TransactionId::INVALID)?,
            pre_total_at_100,
            "total_length_at(100) must be preserved"
        );
        Ok(())
    }

    #[test]
    fn persistence_representability_rejects_custom_tokenizer_and_mvcc_history() {
        let mut current = InvertedIndex::new(BM25Config::default());
        current.insert(NodeId::new(1), "current document");
        assert!(current.is_current_image_persistence_representable());

        let custom = InvertedIndex::with_tokenizer(
            BM25Config::default(),
            Box::new(SimpleTokenizer::with_min_length(7)),
        );
        assert!(!custom.is_current_image_persistence_representable());

        current.insert_versioned(NodeId::new(1), "updated document", EpochId::new(2), None);
        assert!(!current.is_current_image_persistence_representable());
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn exact_runtime_fork_preserves_opaque_tokenizer_history_and_scope()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::graph::write_permit::{WriteAuthority, with_authority};

        let mut source = InvertedIndex::with_tokenizer(
            BM25Config { k1: 1.8, b: 0.2 },
            Box::new(SimpleTokenizer::with_min_length(7)),
        );
        let node = NodeId::new(31);
        source.insert_versioned(node, "tiny enormous enormous", EpochId::new(1), None);
        source.insert_versioned(node, "gigantic gigantic", EpochId::new(3), None);

        let authority = WriteAuthority::new();
        let transition = InvertedIndex::pin_scope_transition();
        assert!(source.bind_under_transition(41, 7, &transition));
        assert!(source.seal_with_scope_under_transition(authority.scope().get(), &transition));
        let mut fork = source.exact_runtime_fork();

        assert!(Arc::ptr_eq(&source.tokenizer, &fork.tokenizer));
        assert_eq!(fork.owner_store.load(Ordering::Acquire), 41);
        assert_eq!(fork.owner_slot.load(Ordering::Acquire), 7);
        assert_eq!(
            fork.mutation_scope.load(Ordering::Acquire),
            authority.scope().get()
        );
        assert_eq!(fork.config().k1.to_bits(), source.config().k1.to_bits());
        for epoch in 0..=4 {
            let epoch = EpochId::new(epoch);
            assert_eq!(
                fork.total_length_at(epoch, TransactionId::INVALID)?,
                source.total_length_at(epoch, TransactionId::INVALID)?
            );
            assert_eq!(
                fork.score_document_visible(
                    node,
                    "tiny enormous gigantic",
                    epoch,
                    TransactionId::INVALID,
                    None,
                    false,
                )?
                .map(f64::to_bits),
                source
                    .score_document_visible(
                        node,
                        "tiny enormous gigantic",
                        epoch,
                        TransactionId::INVALID,
                        None,
                        false,
                    )?
                    .map(f64::to_bits)
            );
        }

        fork.insert(NodeId::new(32), "unauthorized mutation");
        assert!(!fork.contains(NodeId::new(32)));
        with_authority(&authority, || {
            fork.insert(NodeId::new(32), "authorized mutation");
        });
        assert!(fork.contains(NodeId::new(32)));
        assert!(!source.contains(NodeId::new(32)));
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn exact_identity_purge_removes_history_and_allows_clean_same_id_recreation()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        let rolled_back = NodeId::new(41);
        let retained = NodeId::new(42);
        let e1 = EpochId::new(1);
        let e2 = EpochId::new(2);
        let e3 = EpochId::new(3);

        index.insert_versioned(rolled_back, "obsolete shared", e1, None);
        assert!(index.remove_versioned(rolled_back, e2, None));
        index.insert_versioned(rolled_back, "replacement shared", e3, None);
        index.insert_versioned(retained, "retained shared", e1, None);

        assert!(index.purge_identity(rolled_back));
        assert!(!index.doc_lengths.contains_key(&rolled_back));
        assert!(index.postings.values().all(|list| {
            list.postings
                .iter()
                .all(|posting| posting.node_id != rolled_back)
        }));
        assert_eq!(index.doc_count_at(e1, TransactionId::INVALID)?, 1);
        assert_eq!(index.doc_count_at(e3, TransactionId::INVALID)?, 1);
        assert_eq!(index.search("shared", 10)[0].0, retained);

        index.insert_versioned(rolled_back, "fresh identity", EpochId::new(4), None);
        assert_eq!(index.search("fresh", 10)[0].0, rolled_back);
        assert!(index.search("obsolete", 10).is_empty());
        assert!(index.search("replacement", 10).is_empty());
        Ok(())
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn zero_scope_never_seals_or_reports_compatible() {
        let index = InvertedIndex::new(BM25Config::default());
        let transition = InvertedIndex::pin_scope_transition();
        assert!(!index.scope_is_compatible(0, &transition));
        assert!(!index.seal_with_scope_under_transition(0, &transition));
        assert!(index.scope_is_unsealed(&transition));
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn retained_alias_rejects_raw_and_foreign_mutation_after_seal() {
        use crate::graph::write_permit::{WriteAuthority, with_authority};
        use std::sync::Arc;

        let index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        let owner = WriteAuthority::new();
        let foreign = WriteAuthority::new();
        {
            let transition = InvertedIndex::pin_scope_transition();
            assert!(
                index
                    .read()
                    .seal_with_scope_under_transition(owner.scope().get(), &transition)
            );
        }

        index.write().insert(NodeId::new(1), "raw");
        with_authority(&foreign, || {
            index.write().insert(NodeId::new(2), "foreign");
        });
        assert!(index.read().is_empty());

        with_authority(&owner, || {
            index.write().insert(NodeId::new(3), "owned");
        });
        assert!(index.read().contains(NodeId::new(3)));

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_authority(&owner, || {
                index
                    .write()
                    .insert(NodeId::new(4), "committed before panic");
                panic!("injected owner panic");
            });
        }));
        assert!(caught.is_err());
        index.write().insert(NodeId::new(5), "raw after panic");
        assert!(!index.read().contains(NodeId::new(5)));
        assert!(index.read().contains(NodeId::new(4)));
    }

    #[cfg(feature = "lpg")]
    #[test]
    fn mutation_transition_proof_is_retained_and_unwind_safe() {
        use std::time::Duration;

        let index = InvertedIndex::new(BM25Config::default());
        let mutation = index.pin_mutation().expect("unsealed mutation proof");
        // Other tests can briefly transition unrelated indexes. Keep this
        // mutation proof alive while waiting: retaining the global gate here
        // would still make acquisition fail for the entire timeout.
        assert!(
            TEXT_SCOPE_TRANSITION_GATE
                .try_write_for(Duration::from_secs(5))
                .is_some(),
            "ordinary mutation must not retain the global DDL transition gate"
        );
        assert!(
            index.mutation_scope_gate.try_write().is_none(),
            "per-index scope transition must be excluded for the proof's full lifetime"
        );
        drop(mutation);
        assert!(
            index
                .mutation_scope_gate
                .try_write_for(Duration::from_secs(5))
                .is_some(),
            "dropping the proof must release the per-index transition gate"
        );

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _mutation = index.pin_mutation().expect("unsealed mutation proof");
            panic!("injected mutation panic");
        }));
        assert!(caught.is_err());
        assert!(
            index
                .mutation_scope_gate
                .try_write_for(Duration::from_secs(5))
                .is_some(),
            "caught panic must not leak the retained mutation proof"
        );
    }
}
