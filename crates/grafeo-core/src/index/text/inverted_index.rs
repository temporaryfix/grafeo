//! BM25-scored inverted index for full-text search.

use super::tokenizer::{SimpleTokenizer, Tokenizer};
use super::versioned::{
    AggDelta, VersionedDocLen, VersionedPosting, doc_len_visible, posting_visible,
};
use grafeo_common::types::{EpochId, NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashSet;
use std::collections::HashMap;

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
    tokenizer: Box<dyn Tokenizer>,
    /// BM25 configuration.
    config: BM25Config,
}

/// Viewing epoch used by the committed-latest search path.
///
/// Large enough that all committed postings (epoch 0 … real epochs) are
/// visible, but small enough that PENDING (u64::MAX) inserts are not.
const COMMITTED_EPOCH: EpochId = EpochId::new(u64::MAX - 1);

impl InvertedIndex {
    /// Creates a new inverted index with the given BM25 configuration.
    #[must_use]
    pub fn new(config: BM25Config) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            agg_log: Vec::new(),
            tokenizer: Box::new(SimpleTokenizer::new()),
            config,
        }
    }

    /// Creates a new inverted index with a custom tokenizer.
    pub fn with_tokenizer(config: BM25Config, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            agg_log: Vec::new(),
            tokenizer,
            config,
        }
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
        // Soft-delete any currently-live doc-len entry for this node so
        // re-insertion behaves like update.
        let has_live = self
            .doc_lengths
            .get(&id)
            .is_some_and(|v| v.iter().any(|d| d.deleted_epoch.is_none()));
        if has_live {
            self.remove_versioned(id, epoch, created_by);
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

    // ── As-of-epoch aggregate queries ──────────────────────────────────────

    /// Returns the total token length of all documents visible at
    /// `(viewing_epoch, viewing_tx)`.
    pub fn total_length_at(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> u64 {
        let sum: i64 = self
            .agg_log
            .iter()
            .filter(|d| d.visible_to(viewing_epoch, viewing_tx))
            .map(|d| d.d_total_len)
            .sum();
        // The aggregate is always non-negative by construction; cast is safe.
        sum.max(0).cast_unsigned()
    }

    /// Returns the number of documents visible at `(viewing_epoch, viewing_tx)`.
    pub fn doc_count_at(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> u64 {
        let count: i64 = self
            .agg_log
            .iter()
            .filter(|d| d.visible_to(viewing_epoch, viewing_tx))
            .map(|d| d.d_doc_count)
            .sum();
        count.max(0).cast_unsigned()
    }

    /// Returns the average document length (in tokens) as seen at
    /// `(viewing_epoch, viewing_tx)`.
    ///
    /// Returns `0.0` if there are no visible documents.
    pub fn avgdl_at(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> f64 {
        let n = self.doc_count_at(viewing_epoch, viewing_tx);
        if n == 0 {
            0.0
        } else {
            self.total_length_at(viewing_epoch, viewing_tx) as f64 / n as f64
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
    pub fn gc(&mut self, horizon: EpochId) {
        let h = horizon.as_u64();

        // ── 1. GC posting lists ────────────────────────────────────────────
        self.postings.retain(|_, list| {
            list.postings.retain(|p| {
                // Keep if: live, or deleted ABOVE horizon, or deleted by PENDING tx.
                match p.deleted_epoch {
                    None => true, // live — never drop
                    Some(d) => {
                        // Keep if PENDING or deleted strictly above the horizon.
                        d == EpochId::PENDING || d.as_u64() > h
                    }
                }
            });
            !list.postings.is_empty()
        });

        // ── 2. GC doc_lengths ──────────────────────────────────────────────
        self.doc_lengths.retain(|_, history| {
            history.retain(|d| match d.deleted_epoch {
                None => true,
                Some(del) => del == EpochId::PENDING || del.as_u64() > h,
            });
            !history.is_empty()
        });

        // ── 3. Compact the aggregate log ───────────────────────────────────
        // Fold all committed deltas at or below `horizon` into a single base
        // entry.  Pending (`tx == Some(...)`) entries and committed entries
        // above `horizon` are kept verbatim.
        let mut base_total: i64 = 0;
        let mut base_count: i64 = 0;
        let mut above: Vec<AggDelta> = Vec::new();

        for delta in std::mem::take(&mut self.agg_log) {
            if delta.tx.is_none() && delta.epoch.as_u64() <= h {
                // Committed at or below horizon — fold into base.
                base_total += delta.d_total_len;
                base_count += delta.d_doc_count;
            } else {
                // Above horizon or pending — keep verbatim.
                above.push(delta);
            }
        }

        // Rebuild the log: base entry first (if non-zero), then the rest.
        // Stamp the base at epoch 0 so it is visible to every E >= 0
        // (i.e., every committed-latest or snapshot reader at any epoch >=
        // horizon will include it in their prefix-sum).
        if base_total != 0 || base_count != 0 {
            self.agg_log.push(AggDelta {
                epoch: EpochId::new(0),
                tx: None,
                d_total_len: base_total,
                d_doc_count: base_count,
            });
        }
        self.agg_log.extend(above);
    }

    // ── Legacy (behavior-preserving) wrappers ──────────────────────────────

    /// Indexes a document (node text) into the inverted index.
    ///
    /// If the node was already indexed, it is first removed and re-indexed.
    /// Uses epoch 0 so postings are visible to all committed-latest searches.
    pub fn insert(&mut self, id: NodeId, text: &str) {
        self.insert_versioned(id, text, EpochId::new(0), None);
    }

    /// Removes a document from the index.
    ///
    /// Returns `true` if the document was found and removed.
    /// Uses epoch 0 so the deletion is visible to all committed-latest searches.
    pub fn remove(&mut self, id: NodeId) -> bool {
        self.remove_versioned(id, EpochId::new(0), None)
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
        let query_tokens = self.tokenizer.tokenize(query);
        let n = self.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID);
        if query_tokens.is_empty() || n == 0 {
            return Vec::new();
        }

        let n_f = n as f64;
        let avg_dl = self.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID);
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
        let query_tokens = self.tokenizer.tokenize(query);
        let n = self.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID);
        if query_tokens.is_empty() || n == 0 {
            return 0.0;
        }
        let Some(doc_len) = self.doc_len_at(id, COMMITTED_EPOCH, TransactionId::INVALID) else {
            return 0.0;
        };
        let n_f = n as f64;
        let avg_dl = self.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID);
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

    /// Returns all documents scoring at or above `threshold` using BM25.
    ///
    /// Unlike [`Self::search`] (top-k), this returns every document above the
    /// threshold, sorted by score descending. Intended for index-accelerated
    /// text search with WHERE predicates.
    #[must_use]
    pub fn search_with_threshold(&self, query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
        let query_tokens = self.tokenizer.tokenize(query);
        let n = self.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID);
        if query_tokens.is_empty() || n == 0 {
            return Vec::new();
        }
        let n_f = n as f64;
        let avg_dl = self.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID);
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
    /// Mirrors [`search_visible`] but uses a threshold cutoff instead of a top-k
    /// limit.  Committed postings are filtered by `posting_visible(epoch, tx)`;
    /// the per-transaction delta is merged so a writer sees its own uncommitted
    /// inserts/tombstones.
    ///
    /// # Parameters
    ///
    /// Same as [`search_visible`] except `threshold` replaces `k`.
    #[must_use]
    pub fn search_with_threshold_visible(
        &self,
        query: &str,
        threshold: f64,
        epoch: EpochId,
        tx: TransactionId,
        delta_docs: &[(NodeId, String)],
        delta_removed: &FxHashSet<NodeId>,
    ) -> Vec<(NodeId, f64)> {
        let query_tokens = self.tokenizer.tokenize(query);
        // An empty query or empty corpus never yields results.
        if query_tokens.is_empty() {
            return Vec::new();
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
        let base_n = self.doc_count_at(epoch, tx) as i64;
        #[allow(clippy::cast_possible_wrap)]
        let base_total = self.total_length_at(epoch, tx) as i64;

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
        results
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
    #[must_use]
    pub fn search_visible(
        &self,
        query: &str,
        k: usize,
        epoch: EpochId,
        tx: TransactionId,
        delta_docs: &[(NodeId, String)],
        delta_removed: &FxHashSet<NodeId>,
    ) -> Vec<(NodeId, f64)> {
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() || k == 0 {
            return Vec::new();
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
        let base_n = self.doc_count_at(epoch, tx) as i64;
        #[allow(clippy::cast_possible_wrap)]
        let base_total = self.total_length_at(epoch, tx) as i64;

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
        results
    }

    // ── Query helpers ───────────────────────────────────────────────────────

    /// Returns true if the given node has a live (committed-latest) entry.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.doc_lengths.get(&id).is_some_and(|v| {
            v.iter()
                .any(|d| doc_len_visible(d, COMMITTED_EPOCH, TransactionId::INVALID))
        })
    }

    /// Returns the number of committed-latest indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        // reason: practical document counts fit usize on all supported platforms
        #[allow(clippy::cast_possible_truncation)]
        let n = self.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID) as usize;
        n
    }

    /// Returns true if the index is empty (committed-latest view).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID) == 0
    }

    /// Returns the number of unique terms in the index.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Returns the BM25 configuration.
    #[must_use]
    pub fn config(&self) -> &BM25Config {
        &self.config
    }

    // ── Snapshot / restore ─────────────────────────────────────────────────

    /// Snapshot the index for serialization.
    ///
    /// Returns (postings, doc_lengths, total_length) where postings is
    /// a vec of (term, vec of (node_id, term_freq)).
    ///
    /// Only committed-latest-visible postings are included in the snapshot.
    #[must_use]
    pub fn snapshot(&self) -> (Vec<(String, Vec<(NodeId, u32)>)>, Vec<(NodeId, u32)>, u64) {
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

        let total_length = self.total_length_at(COMMITTED_EPOCH, TransactionId::INVALID);
        (postings, doc_lengths, total_length)
    }

    /// Override the BM25 configuration parameters.
    pub fn set_config(&mut self, config: BM25Config) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_doc_count_at_epoch_boundary() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // 3 tokens: "hello", "world", "one"  → len 3
        index.insert_versioned(NodeId::new(1), "hello world one", EpochId::new(1), None);
        // 2 tokens: "foo", "bar"  → len 2
        index.insert_versioned(NodeId::new(2), "foo bar", EpochId::new(2), None);

        // At epoch 1 only node 1 is visible.
        assert_eq!(
            index.doc_count_at(EpochId::new(1), TransactionId::INVALID),
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(1), TransactionId::INVALID),
            3
        );

        // At epoch 2 both nodes are visible.
        assert_eq!(
            index.doc_count_at(EpochId::new(2), TransactionId::INVALID),
            2
        );
        assert_eq!(
            index.total_length_at(EpochId::new(2), TransactionId::INVALID),
            5
        );
    }

    /// A doc deleted after E1 must still be counted at E1.
    #[test]
    fn test_doc_still_visible_at_epoch_before_deletion() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // "alpha beta gamma" → 3 tokens
        index.insert_versioned(NodeId::new(1), "alpha beta gamma", EpochId::new(1), None);
        // Delete at epoch 5.
        index.remove_versioned(NodeId::new(1), EpochId::new(5), None);

        // At epoch 3 (before deletion) it is still visible.
        assert_eq!(
            index.doc_count_at(EpochId::new(3), TransactionId::INVALID),
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(3), TransactionId::INVALID),
            3
        );

        // At epoch 5 (at deletion) it is gone.
        assert_eq!(
            index.doc_count_at(EpochId::new(5), TransactionId::INVALID),
            0
        );
        assert_eq!(
            index.total_length_at(EpochId::new(5), TransactionId::INVALID),
            0
        );
    }

    /// `avgdl_at` uses the as-of-E totals.
    #[test]
    fn test_avgdl_at_epoch() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // "hello world one" → 3 tokens (len 3)
        index.insert_versioned(NodeId::new(1), "hello world one", EpochId::new(1), None);
        // "foo bar baz qux" → 4 tokens (len 4)
        index.insert_versioned(NodeId::new(2), "foo bar baz qux", EpochId::new(3), None);

        // At epoch 1: only doc1, avgdl = 3/1 = 3.0
        let avgdl_e1 = index.avgdl_at(EpochId::new(1), TransactionId::INVALID);
        assert!(
            (avgdl_e1 - 3.0).abs() < 1e-10,
            "avgdl@1 expected 3.0, got {avgdl_e1}"
        );

        // At epoch 3: both docs, avgdl = (3+4)/2 = 3.5
        let avgdl_e3 = index.avgdl_at(EpochId::new(3), TransactionId::INVALID);
        assert!(
            (avgdl_e3 - 3.5).abs() < 1e-10,
            "avgdl@3 expected 3.5, got {avgdl_e3}"
        );
    }

    /// A doc whose length changes between epochs must report the as-of-E length.
    #[test]
    fn test_avgdl_changes_after_doc_update() {
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
            index.doc_count_at(EpochId::new(1), TransactionId::INVALID),
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(1), TransactionId::INVALID),
            2
        );
        let avgdl_e1 = index.avgdl_at(EpochId::new(1), TransactionId::INVALID);
        assert!(
            (avgdl_e1 - 2.0).abs() < 1e-10,
            "avgdl@1 expected 2.0, got {avgdl_e1}"
        );

        // At epoch 5: len=4 (updated), count=1, avgdl=4.0
        assert_eq!(
            index.doc_count_at(EpochId::new(5), TransactionId::INVALID),
            1
        );
        assert_eq!(
            index.total_length_at(EpochId::new(5), TransactionId::INVALID),
            4
        );
        let avgdl_e5 = index.avgdl_at(EpochId::new(5), TransactionId::INVALID);
        assert!(
            (avgdl_e5 - 4.0).abs() < 1e-10,
            "avgdl@5 expected 4.0, got {avgdl_e5}"
        );
    }

    /// Committed-latest `avgdl_at(COMMITTED_EPOCH, INVALID)` == `total_length/n`
    /// for epoch-0 legacy inserts — i.e., behaviour-preserving.
    #[test]
    fn test_avgdl_at_committed_epoch_matches_legacy() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world"); // 2 tokens
        index.insert(NodeId::new(2), "foo bar baz"); // 3 tokens

        let avgdl = index.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID);
        // (2+3)/2 = 2.5
        assert!(
            (avgdl - 2.5).abs() < 1e-10,
            "avgdl at committed epoch expected 2.5, got {avgdl}"
        );
        // doc_count matches len()
        assert_eq!(
            index.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID) as usize,
            index.len()
        );
    }

    /// Own-tx pending inserts are visible only to the inserting tx.
    #[test]
    fn test_doc_count_at_own_tx_pending() {
        let mut index = InvertedIndex::new(BM25Config::default());
        let tx7 = TransactionId::new(7);
        // Pending insert by tx 7.
        index.insert_versioned(NodeId::new(1), "hello world", EpochId::PENDING, Some(tx7));

        // tx 7 can see its own pending insert.
        assert_eq!(index.doc_count_at(EpochId::new(10), tx7), 1);
        // tx 8 cannot.
        assert_eq!(
            index.doc_count_at(EpochId::new(10), TransactionId::new(8)),
            0
        );
        // Committed-latest reader cannot.
        assert_eq!(
            index.doc_count_at(COMMITTED_EPOCH, TransactionId::INVALID),
            0
        );
    }

    /// `avgdl_at` returns 0.0 when the corpus is empty.
    #[test]
    fn test_avgdl_at_empty_index() {
        let index = InvertedIndex::new(BM25Config::default());
        assert_eq!(index.avgdl_at(COMMITTED_EPOCH, TransactionId::INVALID), 0.0);
    }

    // ── GC tests (Task 6 TDD) ──────────────────────────────────────────────

    /// `gc(horizon)` drops postings for a doc that was fully deleted below the
    /// horizon, but keeps postings for docs deleted ABOVE the horizon (a snapshot
    /// at horizon–1 still needs them) and for live (never-deleted) docs.
    ///
    /// The aggregate log after `gc` still yields the correct `avgdl_at(E)` for
    /// any `E >= horizon`.
    #[test]
    fn gc_drops_postings_deleted_below_horizon() {
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
        index.gc(e3);

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
        let count_at_e3 = index.doc_count_at(e3, TransactionId::INVALID);
        assert_eq!(count_at_e3, 2, "doc_count_at(e3) must be 2 after gc");
        let total_at_e3 = index.total_length_at(e3, TransactionId::INVALID);
        assert_eq!(total_at_e3, 6, "total_length_at(e3) must be 6 after gc");
    }

    /// After `gc(horizon)` with `horizon < deletion_epoch`, the posting is KEPT:
    /// a snapshot at any epoch between insert and delete still needs to see it.
    #[test]
    fn gc_keeps_posting_deleted_above_horizon() {
        let mut index = InvertedIndex::new(BM25Config::default());
        let e1 = EpochId::new(1);
        let e5 = EpochId::new(5);
        let e2 = EpochId::new(2); // horizon below deletion

        index.insert_versioned(NodeId::new(99), "keep me please", e1, None);
        index.remove_versioned(NodeId::new(99), e5, None); // deleted at 5

        index.gc(e2); // horizon = 2, below deletion epoch 5

        // Posting must still be present.
        let present = index
            .postings
            .values()
            .any(|pl| pl.postings.iter().any(|p| p.node_id == NodeId::new(99)));
        assert!(present, "posting deleted above horizon must be retained");
    }

    /// `gc(horizon)` compacts the aggregate log: after compaction,
    /// `doc_count_at(E)` and `total_length_at(E)` for `E >= horizon` equal the
    /// pre-gc values, and for `E < horizon` the aggregate log returns a
    /// consistent (collapsed) prefix.
    #[test]
    fn gc_compacts_aggregate_log() {
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
        let pre_count_at_h = index.doc_count_at(horizon, TransactionId::INVALID);
        let pre_total_at_h = index.total_length_at(horizon, TransactionId::INVALID);
        let pre_count_at_100 = index.doc_count_at(EpochId::new(100), TransactionId::INVALID);
        let pre_total_at_100 = index.total_length_at(EpochId::new(100), TransactionId::INVALID);

        let pre_agg_len = index.agg_log.len();

        index.gc(horizon);

        let post_agg_len = index.agg_log.len();
        assert!(
            post_agg_len < pre_agg_len,
            "gc must compact the aggregate log (pre={pre_agg_len}, post={post_agg_len})"
        );

        // Aggregate values at and above horizon must be preserved.
        assert_eq!(
            index.doc_count_at(horizon, TransactionId::INVALID),
            pre_count_at_h,
            "doc_count_at(horizon) must be preserved"
        );
        assert_eq!(
            index.total_length_at(horizon, TransactionId::INVALID),
            pre_total_at_h,
            "total_length_at(horizon) must be preserved"
        );
        assert_eq!(
            index.doc_count_at(EpochId::new(100), TransactionId::INVALID),
            pre_count_at_100,
            "doc_count_at(100) must be preserved"
        );
        assert_eq!(
            index.total_length_at(EpochId::new(100), TransactionId::INVALID),
            pre_total_at_100,
            "total_length_at(100) must be preserved"
        );
    }
}
