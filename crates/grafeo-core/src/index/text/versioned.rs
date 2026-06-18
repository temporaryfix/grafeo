//! Versioned posting types for snapshot-isolated BM25 text search.
//!
//! Each posting carries MVCC visibility metadata so it can be filtered to a
//! transaction's snapshot epoch.  The logic mirrors [`VersionInfo::is_visible_to`]
//! in `grafeo-common::mvcc` exactly.

use grafeo_common::types::{EpochId, NodeId, TransactionId};

// ── VersionedDocLen ─────────────────────────────────────────────────────────

/// A versioned record of a document's token length.
///
/// Mirrors the MVCC visibility shape of [`VersionedPosting`]: a document's
/// length at a given epoch is the entry visible at that epoch.  When a
/// document is re-inserted (updated) the old entry is soft-deleted and a new
/// one is appended, so per-doc length history is preserved.
#[derive(Debug, Clone)]
pub(super) struct VersionedDocLen {
    pub(super) len: u32,
    pub(super) created_epoch: EpochId,
    /// `Some(tx)` while the creating transaction is uncommitted.
    pub(super) created_by: Option<TransactionId>,
    /// `None` = live; `Some(E)` = deleted at epoch `E`.
    pub(super) deleted_epoch: Option<EpochId>,
    /// `Some(tx)` while the deleting transaction is uncommitted.
    pub(super) deleted_by: Option<TransactionId>,
}

impl VersionedDocLen {
    /// Constructs a new live doc-length record stamped with the given epoch.
    pub(super) fn new(len: u32, created_epoch: EpochId, created_by: Option<TransactionId>) -> Self {
        Self {
            len,
            created_epoch,
            created_by,
            deleted_epoch: None,
            deleted_by: None,
        }
    }
}

/// Returns `true` if the [`VersionedDocLen`] entry is visible to a reader at
/// `(viewing_epoch, viewing_tx)`.
///
/// Identical visibility rules as [`posting_visible`].
#[inline]
pub(super) fn doc_len_visible(
    d: &VersionedDocLen,
    viewing_epoch: EpochId,
    viewing_tx: TransactionId,
) -> bool {
    if d.deleted_by == Some(viewing_tx) {
        return false;
    }
    if d.created_by == Some(viewing_tx) {
        return d.deleted_epoch.is_none();
    }
    if d.created_epoch.as_u64() > viewing_epoch.as_u64() {
        return false;
    }
    if let Some(del) = d.deleted_epoch {
        del.as_u64() > viewing_epoch.as_u64()
    } else {
        true
    }
}

// ── AggDelta ────────────────────────────────────────────────────────────────

/// One entry in the epoch-stamped aggregate log.
///
/// Appended on every `insert_versioned` / `remove_versioned` call so that
/// `total_length@E` and `doc_count@E` can be reconstructed by a prefix-sum
/// over entries with `delta.epoch <= E`.
///
/// For **pending** (uncommitted) operations the epoch is [`EpochId::PENDING`]
/// and the `tx` field carries the owning transaction.  Such deltas are
/// included in a prefix-sum only when the caller's `viewing_tx` matches.
#[derive(Debug, Clone)]
pub(super) struct AggDelta {
    pub(super) epoch: EpochId,
    /// Owning transaction for pending deltas; `None` for committed ones.
    pub(super) tx: Option<TransactionId>,
    pub(super) d_total_len: i64,
    pub(super) d_doc_count: i64,
}

impl AggDelta {
    /// Returns `true` if this delta should be counted by a reader at
    /// `(viewing_epoch, viewing_tx)`.
    ///
    /// Committed delta (`tx == None`): count iff `delta.epoch <= viewing_epoch`.
    /// Pending delta (`tx == Some(t)`): count iff `viewing_tx == t`.
    #[inline]
    pub(super) fn visible_to(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> bool {
        match self.tx {
            None => self.epoch.as_u64() <= viewing_epoch.as_u64(),
            Some(t) => t == viewing_tx,
        }
    }
}

// ── VersionedPosting ────────────────────────────────────────────────────────

/// A posting entry with MVCC visibility metadata.
///
/// `created_by` is `Some(tx)` while the creating transaction is still
/// uncommitted (`created_epoch == EpochId::PENDING`).  It becomes `None`
/// after commit / for legacy epoch-0 inserts.
///
/// `deleted_epoch` / `deleted_by` follow the same convention: `None` means
/// live; `Some(EpochId::PENDING)` means deleted by an uncommitted transaction;
/// `Some(E)` means the deletion was committed at epoch `E`.
#[derive(Debug, Clone)]
pub(super) struct VersionedPosting {
    pub(super) node_id: NodeId,
    pub(super) term_freq: u32,
    pub(super) created_epoch: EpochId,
    /// `Some(tx)` while the creating transaction is uncommitted.
    pub(super) created_by: Option<TransactionId>,
    /// `None` = live; `Some(E)` = deleted at epoch `E`.
    pub(super) deleted_epoch: Option<EpochId>,
    /// `Some(tx)` while the deleting transaction is uncommitted.
    pub(super) deleted_by: Option<TransactionId>,
}

impl VersionedPosting {
    /// Constructs a new live posting stamped with the given epoch.
    pub(super) fn new(
        node_id: NodeId,
        term_freq: u32,
        created_epoch: EpochId,
        created_by: Option<TransactionId>,
    ) -> Self {
        Self {
            node_id,
            term_freq,
            created_epoch,
            created_by,
            deleted_epoch: None,
            deleted_by: None,
        }
    }
}

// ── Visibility ─────────────────────────────────────────────────────────────

/// Returns `true` if `posting` is visible to a reader at `(viewing_epoch, viewing_tx)`.
///
/// Mirrors [`VersionInfo::is_visible_to`] exactly:
/// 1. Own-tx delete → not visible.
/// 2. Own-tx create (and not own-deleted) → visible.
/// 3. Otherwise: epoch-based — created at or before `viewing_epoch` and not
///    deleted at or before `viewing_epoch`.
///
/// For committed-latest callers that have no transaction identity pass
/// `viewing_tx = TransactionId::INVALID`; the own-tx branches will never fire.
#[inline]
pub(super) fn posting_visible(
    p: &VersionedPosting,
    viewing_epoch: EpochId,
    viewing_tx: TransactionId,
) -> bool {
    // 1. If this posting was deleted by the viewing transaction → invisible.
    if p.deleted_by == Some(viewing_tx) {
        return false;
    }

    // 2. Own-tx creation: visible iff not deleted.
    if p.created_by == Some(viewing_tx) {
        return p.deleted_epoch.is_none();
    }

    // 3. Epoch-based visibility.
    //    Created at or before viewing_epoch …
    if p.created_epoch.as_u64() > viewing_epoch.as_u64() {
        return false;
    }
    //    … and not deleted at or before viewing_epoch.
    if let Some(del) = p.deleted_epoch {
        del.as_u64() > viewing_epoch.as_u64()
    } else {
        true
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn posting(
        created_epoch: u64,
        created_by: Option<u64>,
        deleted_epoch: Option<u64>,
        deleted_by: Option<u64>,
    ) -> VersionedPosting {
        VersionedPosting {
            node_id: NodeId::new(1),
            term_freq: 1,
            created_epoch: EpochId::new(created_epoch),
            created_by: created_by.map(TransactionId::new),
            deleted_epoch: deleted_epoch.map(EpochId::new),
            deleted_by: deleted_by.map(TransactionId::new),
        }
    }

    fn visible(p: &VersionedPosting, epoch: u64, tx: u64) -> bool {
        posting_visible(p, EpochId::new(epoch), TransactionId::new(tx))
    }

    // ── Case A: live posting created before viewing epoch ──────────────────

    #[test]
    fn live_posting_visible_at_creation_epoch() {
        let p = posting(5, None, None, None);
        assert!(visible(&p, 5, TransactionId::INVALID.0));
    }

    #[test]
    fn live_posting_visible_after_creation_epoch() {
        let p = posting(5, None, None, None);
        assert!(visible(&p, 10, TransactionId::INVALID.0));
    }

    #[test]
    fn live_posting_invisible_before_creation_epoch() {
        let p = posting(5, None, None, None);
        assert!(!visible(&p, 4, TransactionId::INVALID.0));
    }

    // ── Case B: posting deleted before viewing epoch ───────────────────────

    #[test]
    fn deleted_posting_invisible_at_deletion_epoch() {
        // deleted at epoch 10 → invisible at epoch 10 (boundary: <= )
        let p = posting(5, None, Some(10), None);
        assert!(!visible(&p, 10, TransactionId::INVALID.0));
    }

    #[test]
    fn deleted_posting_invisible_after_deletion_epoch() {
        let p = posting(5, None, Some(10), None);
        assert!(!visible(&p, 15, TransactionId::INVALID.0));
    }

    #[test]
    fn deleted_posting_visible_before_deletion_epoch() {
        let p = posting(5, None, Some(10), None);
        assert!(visible(&p, 9, TransactionId::INVALID.0));
    }

    // ── Case C: own-tx pending create ─────────────────────────────────────

    #[test]
    fn own_pending_create_visible_to_creating_tx() {
        // created_epoch = PENDING, created_by = tx 7
        let p = posting(EpochId::PENDING.0, Some(7), None, None);
        assert!(visible(&p, 5, 7));
    }

    #[test]
    fn own_pending_create_invisible_to_other_tx() {
        let p = posting(EpochId::PENDING.0, Some(7), None, None);
        // PENDING epoch > any real epoch → epoch check fails for other tx
        assert!(!visible(&p, 5, 8));
        assert!(!visible(&p, 5, TransactionId::INVALID.0));
    }

    // ── Case D: own-tx pending delete ─────────────────────────────────────

    #[test]
    fn own_pending_delete_invisible_to_deleting_tx() {
        // created at epoch 1 by someone else; now pending-deleted by tx 7
        let p = posting(1, None, Some(EpochId::PENDING.0), Some(7));
        assert!(!visible(&p, 5, 7));
    }

    #[test]
    fn own_pending_delete_visible_to_other_tx() {
        let p = posting(1, None, Some(EpochId::PENDING.0), Some(7));
        // Other tx at epoch 5: deleted_by != their tx, created_by != their tx,
        // created_epoch=1 <= 5, deleted_epoch=PENDING > 5 → visible.
        assert!(visible(&p, 5, 8));
        // Also visible to epoch-only callers.
        assert!(visible(&p, 5, TransactionId::INVALID.0));
    }

    // ── Case E: legacy insert/remove paths (epoch 0) ───────────────────────

    #[test]
    fn legacy_insert_epoch0_always_visible() {
        let p = posting(0, None, None, None);
        // Visible at epoch 0 and beyond.
        assert!(visible(&p, 0, TransactionId::INVALID.0));
        assert!(visible(&p, 1_000, TransactionId::INVALID.0));
    }

    #[test]
    fn legacy_remove_epoch0_always_invisible() {
        // Legacy remove stamps deleted_epoch=Some(0), deleted_by=None.
        let p = posting(0, None, Some(0), None);
        // At epoch 0: created_epoch=0 <=0, deleted_epoch=0 > 0 → false → invisible.
        assert!(!visible(&p, 0, TransactionId::INVALID.0));
        // At any later epoch the deleted_epoch=0 <= viewing_epoch → invisible.
        assert!(!visible(&p, 5, TransactionId::INVALID.0));
        assert!(!visible(&p, 1_000, TransactionId::INVALID.0));
    }

    // ── VersionedDocLen visibility tests ──────────────────────────────────

    fn doc_len(
        len: u32,
        created_epoch: u64,
        created_by: Option<u64>,
        deleted_epoch: Option<u64>,
        deleted_by: Option<u64>,
    ) -> VersionedDocLen {
        VersionedDocLen {
            len,
            created_epoch: EpochId::new(created_epoch),
            created_by: created_by.map(TransactionId::new),
            deleted_epoch: deleted_epoch.map(EpochId::new),
            deleted_by: deleted_by.map(TransactionId::new),
        }
    }

    fn dl_visible(d: &VersionedDocLen, epoch: u64, tx: u64) -> bool {
        doc_len_visible(d, EpochId::new(epoch), TransactionId::new(tx))
    }

    #[test]
    fn doc_len_live_visible_at_creation_epoch() {
        let d = doc_len(4, 5, None, None, None);
        assert!(dl_visible(&d, 5, TransactionId::INVALID.0));
        assert!(dl_visible(&d, 10, TransactionId::INVALID.0));
    }

    #[test]
    fn doc_len_live_invisible_before_creation_epoch() {
        let d = doc_len(4, 5, None, None, None);
        assert!(!dl_visible(&d, 4, TransactionId::INVALID.0));
    }

    #[test]
    fn doc_len_deleted_invisible_at_and_after_deletion() {
        let d = doc_len(4, 3, None, Some(7), None);
        assert!(!dl_visible(&d, 7, TransactionId::INVALID.0));
        assert!(!dl_visible(&d, 10, TransactionId::INVALID.0));
        // but visible before deletion
        assert!(dl_visible(&d, 6, TransactionId::INVALID.0));
    }

    #[test]
    fn doc_len_own_tx_pending_create_visible_to_creating_tx() {
        let d = doc_len(4, EpochId::PENDING.0, Some(7), None, None);
        assert!(dl_visible(&d, 5, 7));
        assert!(!dl_visible(&d, 5, 8));
        assert!(!dl_visible(&d, 5, TransactionId::INVALID.0));
    }

    #[test]
    fn doc_len_own_tx_pending_delete_invisible_to_deleting_tx() {
        let d = doc_len(4, 1, None, Some(EpochId::PENDING.0), Some(7));
        assert!(!dl_visible(&d, 5, 7));
        assert!(dl_visible(&d, 5, 8));
        assert!(dl_visible(&d, 5, TransactionId::INVALID.0));
    }

    // ── AggDelta visibility tests ──────────────────────────────────────────

    #[test]
    fn agg_delta_committed_visible_at_and_after_epoch() {
        let d = AggDelta {
            epoch: EpochId::new(5),
            tx: None,
            d_total_len: 10,
            d_doc_count: 1,
        };
        assert!(d.visible_to(EpochId::new(5), TransactionId::INVALID));
        assert!(d.visible_to(EpochId::new(10), TransactionId::INVALID));
        assert!(!d.visible_to(EpochId::new(4), TransactionId::INVALID));
    }

    #[test]
    fn agg_delta_pending_visible_only_to_own_tx() {
        let d = AggDelta {
            epoch: EpochId::PENDING,
            tx: Some(TransactionId::new(7)),
            d_total_len: 10,
            d_doc_count: 1,
        };
        assert!(d.visible_to(EpochId::new(5), TransactionId::new(7)));
        assert!(!d.visible_to(EpochId::new(5), TransactionId::new(8)));
        assert!(!d.visible_to(EpochId::new(5), TransactionId::INVALID));
    }
}
