//! Sparse, final-row Text maintenance. Recorded inputs use the same installer.

use super::{
    AggDelta, BM25Config, ExactTokenizerDescriptor, InvertedIndex, PostingList, Tokenizer,
    VersionedDocLen, VersionedPosting,
};
use crate::graph::lpg::DataRebindError;
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::Result;
use parking_lot::{ArcRwLockWriteGuard, RawRwLock, RwLock};
use std::sync::Arc;

type RebindResult<T> = std::result::Result<T, DataRebindError>;

mod wal;

struct DocumentChange {
    id: NodeId,
    input: usize,
    tokens: Vec<String>,
    new_len: u32,
    old_len: Option<u32>,
    expected_len: usize,
    existed: bool,
    close: Option<(usize, VersionedDocLen)>,
    insert_at: usize,
    additions: Vec<VersionedDocLen>,
    posting_total: u64,
    last_term: Option<usize>,
}

struct TermChange {
    term: String,
    existed: bool,
    expected_len: usize,
    insert_at: usize,
    close: Vec<(usize, VersionedPosting)>,
    additions: PostingList,
}

/// Outer owner of raw input, completed and partial fragments, and anchors.
///
/// Keep this outside enclosing publication gates. Installation retains old
/// histories in place and transfers only new missing-map payloads; temporary
/// token/key/append buffers stay here until those enclosing gates have drained.
/// Discovery scans existing postings because Text has no document-to-term
/// directory. Scratch is proportional to affected postings and input tokens,
/// not unrelated posting-list or index history lengths.
pub(crate) struct TextCommitWorkspace {
    inputs: Vec<(NodeId, Option<String>)>,
    rebuild_configuration: Option<(BM25Config, usize)>,
    supplied_documents: Vec<NodeId>,
    frontier: EpochId,
    commit_epoch: EpochId,
    transaction_id: TransactionId,
    order: Vec<usize>,
    documents: Vec<DocumentChange>,
    terms: Vec<TermChange>,
    token_order: Vec<(usize, usize)>,
    aggregate: Vec<AggDelta>,
    aggregate_len: usize,
    aggregate_insert_at: usize,
    aggregate_total: i64,
    aggregate_count: i64,
    wal_header: Option<wal::Header>,
    recorded: Option<Vec<u8>>,
    missing_documents: usize,
    missing_terms: usize,
    gate: Option<Arc<RwLock<()>>>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    scope_attempted: bool,
    preparation_attempted: bool,
    prepared: bool,
}

impl TextCommitWorkspace {
    pub(crate) fn new(
        changes: Vec<(NodeId, Option<String>)>,
        frontier: EpochId,
        commit_epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Self {
        Self {
            inputs: changes,
            rebuild_configuration: None,
            supplied_documents: Vec::new(),
            frontier,
            commit_epoch,
            transaction_id,
            order: Vec::new(),
            documents: Vec::new(),
            terms: Vec::new(),
            token_order: Vec::new(),
            aggregate: Vec::new(),
            aggregate_len: 0,
            aggregate_insert_at: 0,
            aggregate_total: 0,
            aggregate_count: 0,
            wal_header: None,
            recorded: None,
            missing_documents: 0,
            missing_terms: 0,
            gate: None,
            tokenizer: None,
            scope_attempted: false,
            preparation_attempted: false,
            prepared: false,
        }
    }

    pub(crate) fn for_rebuild(
        changes: Vec<(NodeId, Option<String>)>,
        frontier: EpochId,
        commit_epoch: EpochId,
        transaction_id: TransactionId,
        configuration: (BM25Config, usize),
    ) -> Self {
        let mut workspace = Self::new(changes, frontier, commit_epoch, transaction_id);
        workspace.rebuild_configuration = Some(configuration);
        workspace
    }
}

/// Excludes ordinary retained-alias mutation across preparation and release.
///
/// The store transition alone does not provide this exclusion. The workspace
/// retains the gate allocation independently of this owning writer guard.
pub(crate) struct TextCommitScope {
    guard: ArcRwLockWriteGuard<RawRwLock, ()>,
}

impl TextCommitScope {
    pub(crate) fn matches(&self, index: &InvertedIndex) -> bool {
        Arc::ptr_eq(
            ArcRwLockWriteGuard::rwlock(&self.guard),
            &index.mutation_scope_gate,
        ) && index.registry_target.is_none()
    }
}

#[cfg(test)]
pub(crate) struct ReleasedTextCommit<'workspace, 'scope> {
    workspace: &'workspace mut TextCommitWorkspace,
    scope: &'scope TextCommitScope,
}

#[cfg(test)]
pub(crate) struct PreparedTextCommit<'index, 'workspace, 'scope> {
    index: &'index mut InvertedIndex,
    workspace: &'workspace mut TextCommitWorkspace,
    scope: &'scope TextCommitScope,
}

/// Retains the concrete target and scope loans until all aggregate installs end.
#[cfg(test)]
pub(crate) struct InstalledTextCommit<'index, 'workspace, 'scope> {
    _index: &'index mut InvertedIndex,
    _workspace: &'workspace mut TextCommitWorkspace,
    _scope: &'scope TextCommitScope,
}

impl InvertedIndex {
    /// Called under the initial concrete target guard, never under final entity
    /// writers. Alias contention is nonblocking: an alias can already own its
    /// outer writer while waiting for this local gate's shared side.
    pub(crate) fn pin_commit_scope(
        &self,
        workspace: &mut TextCommitWorkspace,
    ) -> Result<TextCommitScope> {
        if workspace.scope_attempted {
            return Err(
                DataRebindError::new("Text commit scope was already attempted").into_error(),
            );
        }
        workspace.scope_attempted = true;
        workspace.gate = Some(Arc::clone(&self.mutation_scope_gate));
        workspace.tokenizer = Some(Arc::clone(&self.tokenizer));
        if self.registry_target.is_some() {
            return Err(DataRebindError::new("Text commit needs a concrete target").into_error());
        }
        self.mutation_scope_gate
            .try_write_arc()
            .map(|guard| TextCommitScope { guard })
            .ok_or_else(|| {
                DataRebindError::Conflict("Text commit mutation scope is in use").into_error()
            })
    }

    /// Constructs all sparse fragments and reserves final destinations before
    /// releasing the initial concrete target writer. No public mutator reentry.
    #[cfg(test)]
    pub(crate) fn prepare_commit<'workspace, 'scope>(
        &mut self,
        workspace: &'workspace mut TextCommitWorkspace,
        scope: &'scope TextCommitScope,
    ) -> Result<ReleasedTextCommit<'workspace, 'scope>> {
        self.prepare_commit_fragments(workspace, scope)?;
        Ok(ReleasedTextCommit { workspace, scope })
    }

    /// The registry retains the exact mutation scope while initial target
    /// writers drain. Final binding uses the collective preallocated fence.
    pub(crate) fn prepare_commit_fragments(
        &mut self,
        workspace: &mut TextCommitWorkspace,
        scope: &TextCommitScope,
    ) -> Result<()> {
        prepare(self, workspace, scope).map_err(DataRebindError::into_error)
    }
}

#[cfg(test)]
impl<'workspace, 'scope> ReleasedTextCommit<'workspace, 'scope> {
    /// Uses the exact concrete target already owned by the collective Text
    /// fence. Does not acquire caller, target or mutation-scope locks.
    pub(crate) fn bind<'index>(
        self,
        index: &'index mut InvertedIndex,
    ) -> RebindResult<PreparedTextCommit<'index, 'workspace, 'scope>> {
        qualify(index, self.workspace, self.scope)?;
        Ok(PreparedTextCommit {
            index,
            workspace: self.workspace,
            scope: self.scope,
        })
    }
}

#[cfg(test)]
impl<'index, 'workspace, 'scope> PreparedTextCommit<'index, 'workspace, 'scope> {
    pub(crate) fn install(self) -> InstalledTextCommit<'index, 'workspace, 'scope> {
        self.workspace.install_prequalified(self.index, self.scope);
        InstalledTextCommit {
            _index: self.index,
            _workspace: self.workspace,
            _scope: self.scope,
        }
    }
}

impl TextCommitWorkspace {
    /// Constituent validation for the aggregate's linear ready proof. Its
    /// collective fence retains both exact target and scope until installation.
    pub(crate) fn validate_prepared(
        &self,
        index: &InvertedIndex,
        scope: &TextCommitScope,
    ) -> RebindResult<()> {
        qualify(index, self, scope)
    }

    /// Only the component ready proof or the already-validated aggregate fence
    /// may call this. Neither owner releases or exposes its exclusive target or
    /// exact scope between validation and install; no post-marker revalidation.
    pub(crate) fn install_prequalified(
        &mut self,
        index: &mut InvertedIndex,
        _scope: &TextCommitScope,
    ) {
        let epoch = self.commit_epoch;
        // Exact retained scope + exclusive concrete target preserve every
        // qualified key, position and capacity. These are prequalified keyed
        // lookups, not late target discovery or fallible publication work.
        for term in &mut self.terms {
            if term.existed {
                if let Some(list) = index.postings.get_mut(&term.term) {
                    for (position, _) in &term.close {
                        if let Some(posting) = list.postings.get_mut(*position) {
                            posting.deleted_epoch = Some(epoch);
                            posting.deleted_by = None;
                        }
                    }
                    append_before(
                        &mut list.postings,
                        &mut term.additions.postings,
                        term.insert_at,
                    );
                }
            } else {
                // Vacant keys/capacity were qualified before durability. Move
                // payload ownership into the live index, leaving no retiree.
                index.postings.insert(
                    std::mem::take(&mut term.term),
                    std::mem::take(&mut term.additions),
                );
            }
        }
        for document in &mut self.documents {
            if document.existed {
                if let Some(history) = index.doc_lengths.get_mut(&document.id) {
                    if let Some((position, _)) = &document.close
                        && let Some(entry) = history.get_mut(*position)
                    {
                        entry.deleted_epoch = Some(epoch);
                        entry.deleted_by = None;
                    }
                    append_before(history, &mut document.additions, document.insert_at);
                }
            } else if !document.additions.is_empty() {
                index
                    .doc_lengths
                    .insert(document.id, std::mem::take(&mut document.additions));
            }
        }
        append_before(
            &mut index.agg_log,
            &mut self.aggregate,
            self.aggregate_insert_at,
        );
        self.prepared = false;
    }
}

fn append_before<T>(target: &mut Vec<T>, additions: &mut Vec<T>, position: usize) {
    let count = additions.len();
    target.append(additions);
    // qualify proved position <= old length and sufficient reserved capacity.
    if let Some(suffix) = target.get_mut(position..) {
        suffix.rotate_right(count);
    }
}

fn reserve<T>(buffer: &mut Vec<T>, additional: usize) -> RebindResult<()> {
    reservation_point()?;
    buffer
        .try_reserve(additional)
        .map_err(|_| AllocError::OutOfMemory.into())
}

fn reservation_point() -> RebindResult<()> {
    #[cfg(test)]
    if FAIL_RESERVATION.with(|point| {
        let remaining = point.get();
        if let Some(remaining) = remaining {
            point.set(remaining.checked_sub(1));
            remaining == 0
        } else {
            false
        }
    }) {
        return Err(AllocError::OutOfMemory.into());
    }
    Ok(())
}

fn prepare(
    index: &mut InvertedIndex,
    workspace: &mut TextCommitWorkspace,
    scope: &TextCommitScope,
) -> RebindResult<()> {
    if workspace.preparation_attempted {
        return Err(DataRebindError::new(
            "Text commit preparation was already attempted",
        ));
    }
    workspace.preparation_attempted = true;
    if !scope.matches(index)
        || !workspace
            .gate
            .as_ref()
            .is_some_and(|gate| Arc::ptr_eq(gate, &index.mutation_scope_gate))
    {
        return Err(DataRebindError::new("Text commit scope identity differs"));
    }
    if workspace.frontier == EpochId::PENDING
        || workspace.commit_epoch == EpochId::PENDING
        || workspace.commit_epoch <= workspace.frontier
        || workspace.transaction_id == TransactionId::INVALID
    {
        return Err(DataRebindError::new(
            "Text commit needs a real transaction and P < C < PENDING",
        ));
    }
    if workspace.frontier < index.retained_from {
        return Err(DataRebindError::Conflict(
            "Text publication frontier is below the retained history floor",
        ));
    }
    if let Some(payload) = workspace.recorded.take() {
        let result = wal::prepare_recorded(index, workspace, &payload);
        workspace.recorded = Some(payload);
        result?;
    } else {
        workspace.wal_header = wal::Header::capture(index);
        if let Some((config, min_token_length)) = &workspace.rebuild_configuration {
            let min_token_length = u64::try_from(*min_token_length).map_err(|_| {
                DataRebindError::new("Text tokenizer length exceeds its exact descriptor")
            })?;
            if config.k1.to_bits() != index.config.k1.to_bits()
                || config.b.to_bits() != index.config.b.to_bits()
                || index.tokenizer_descriptor
                    != Some(ExactTokenizerDescriptor::Simple { min_token_length })
            {
                return Err(DataRebindError::Conflict(
                    "Text rebuild configuration differs from its canonical owner",
                ));
            }
            reserve(&mut workspace.supplied_documents, workspace.inputs.len())?;
            workspace
                .supplied_documents
                .extend(workspace.inputs.iter().map(|(id, _)| *id));
            workspace.supplied_documents.sort_unstable();
            if workspace
                .supplied_documents
                .windows(2)
                .any(|ids| ids[0] == ids[1])
            {
                return Err(DataRebindError::new(
                    "Text rebuild contains duplicate document identities",
                ));
            }
            reserve(&mut workspace.inputs, index.doc_lengths.len())?;
            for id in index.doc_lengths.keys() {
                if workspace.supplied_documents.binary_search(id).is_err() {
                    workspace.inputs.push((*id, None));
                }
            }
        }
        normalize_and_tokenize(index, workspace)?;
        prepare_documents(index, workspace)?;
        prepare_terms(index, workspace)?;
        prepare_aggregate(index, workspace)?;
    }
    reservation_point()?;
    index
        .postings
        .try_reserve(workspace.missing_terms)
        .map_err(|_| AllocError::OutOfMemory)?;
    reservation_point()?;
    index
        .doc_lengths
        .try_reserve(workspace.missing_documents)
        .map_err(|_| AllocError::OutOfMemory)?;
    for term in &workspace.terms {
        if term.existed
            && let Some(list) = index.postings.get_mut(&term.term)
        {
            reserve(&mut list.postings, term.additions.postings.len())?;
        }
    }
    for document in &workspace.documents {
        if document.existed
            && let Some(history) = index.doc_lengths.get_mut(&document.id)
        {
            reserve(history, document.additions.len())?;
        }
    }
    reserve(&mut index.agg_log, workspace.aggregate.len())?;
    workspace.prepared = true;
    qualify(index, workspace, scope)
}

fn normalize_and_tokenize(
    index: &InvertedIndex,
    workspace: &mut TextCommitWorkspace,
) -> RebindResult<()> {
    reserve(&mut workspace.order, workspace.inputs.len())?;
    workspace.order.extend(0..workspace.inputs.len());
    workspace
        .order
        .sort_unstable_by_key(|&position| (workspace.inputs[position].0, position));
    reserve(&mut workspace.documents, workspace.inputs.len())?;
    let mut cursor = 0;
    while cursor < workspace.order.len() {
        let id = workspace.inputs[workspace.order[cursor]].0;
        let mut end = cursor + 1;
        while end < workspace.order.len() && workspace.inputs[workspace.order[end]].0 == id {
            end += 1;
        }
        // Raw duplicate inputs stay outer-owned; the final supplied row wins.
        workspace.documents.push(DocumentChange {
            id,
            input: workspace.order[end - 1],
            tokens: Vec::new(),
            new_len: 0,
            old_len: None,
            expected_len: 0,
            existed: false,
            close: None,
            insert_at: 0,
            additions: Vec::new(),
            posting_total: 0,
            last_term: None,
        });
        cursor = end;
    }
    for (position, document) in workspace.documents.iter_mut().enumerate() {
        if !document.id.is_valid() {
            return Err(DataRebindError::new(
                "Text commit contains an invalid node identity",
            ));
        }
        if let Some(text) = &workspace.inputs[document.input].1 {
            // The returned tokenizer payload is immediately workspace-owned,
            // including when a later document/tokenizer fails or unwinds.
            document.tokens = index.tokenizer.tokenize(text);
        }
        document.new_len = u32::try_from(document.tokens.len())
            .map_err(|_| DataRebindError::new("Text document token count exceeds u32"))?;
        reserve(&mut workspace.token_order, document.tokens.len())?;
        workspace
            .token_order
            .extend((0..document.tokens.len()).map(|token| (position, token)));
        if document.new_len != 0 {
            reserve(&mut document.additions, 1)?;
            document.additions.push(VersionedDocLen::new(
                document.new_len,
                workspace.commit_epoch,
                None,
            ));
        }
    }
    workspace
        .token_order
        .sort_unstable_by(|&(left_doc, left_token), &(right_doc, right_token)| {
            workspace.documents[left_doc].tokens[left_token]
                .cmp(&workspace.documents[right_doc].tokens[right_token])
                .then(left_doc.cmp(&right_doc))
        });
    Ok(())
}

/// Validates only touched histories. Foreign pending creations remain exact;
/// committed histories may not overwrite another transaction's pending delete.
fn committed_live(
    created: EpochId,
    creator: Option<TransactionId>,
    deleted: Option<EpochId>,
    deleter: Option<TransactionId>,
    frontier: EpochId,
    tx: TransactionId,
) -> RebindResult<bool> {
    if creator == Some(tx) || deleter == Some(tx) {
        return Err(DataRebindError::new(
            "Text commit rejects own write-through pending history",
        ));
    }
    if created == EpochId::PENDING {
        if creator.is_none() {
            return Err(DataRebindError::new("Text pending creation has no owner"));
        }
        return Ok(false);
    }
    if creator.is_some() || (deleted.is_none() && deleter.is_some()) {
        return Err(DataRebindError::new(
            "Text committed history has invalid owner metadata",
        ));
    }
    if created > frontier {
        return Err(DataRebindError::Conflict(
            "Text creation is newer than publication frontier",
        ));
    }
    match deleted {
        Some(epoch) if epoch == EpochId::PENDING => {
            if deleter.is_none() {
                return Err(DataRebindError::new("Text pending deletion has no owner"));
            }
            Err(DataRebindError::Conflict(
                "Text touched history has a foreign pending deletion",
            ))
        }
        Some(epoch) => {
            if deleter.is_some() || epoch < created {
                return Err(DataRebindError::new(
                    "Text committed deletion metadata is invalid",
                ));
            }
            if epoch > frontier {
                return Err(DataRebindError::Conflict(
                    "Text deletion is newer than publication frontier",
                ));
            }
            Ok(false)
        }
        None => Ok(true),
    }
}

fn prepare_documents(
    index: &InvertedIndex,
    workspace: &mut TextCommitWorkspace,
) -> RebindResult<()> {
    for document in &mut workspace.documents {
        if let Some(history) = index.doc_lengths.get(&document.id) {
            document.existed = true;
            document.expected_len = history.len();
            document.insert_at = history
                .iter()
                .position(|entry| entry.created_epoch == EpochId::PENDING)
                .map_or(history.len(), |position| position);
            for (position, entry) in history.iter().enumerate() {
                if committed_live(
                    entry.created_epoch,
                    entry.created_by,
                    entry.deleted_epoch,
                    entry.deleted_by,
                    workspace.frontier,
                    workspace.transaction_id,
                )? {
                    if document.close.is_some() {
                        return Err(DataRebindError::new(
                            "Text document has multiple committed live lengths",
                        ));
                    }
                    document.close = Some((position, entry.clone()));
                    document.old_len = Some(entry.len);
                }
            }
        } else if document.new_len != 0 {
            workspace.missing_documents += 1;
        }
    }
    Ok(())
}

fn add_term(
    workspace: &mut TextCommitWorkspace,
    term: &str,
    existing: Option<&PostingList>,
) -> RebindResult<usize> {
    reserve(&mut workspace.terms, 1)?;
    let position = workspace.terms.len();
    workspace.terms.push(TermChange {
        term: String::new(),
        existed: existing.is_some(),
        expected_len: existing.map_or(0, |list| list.postings.len()),
        insert_at: existing.map_or(0, |list| {
            list.postings
                .iter()
                .position(|posting| posting.created_epoch == EpochId::PENDING)
                .map_or(list.postings.len(), |position| position)
        }),
        close: Vec::new(),
        additions: PostingList::default(),
    });
    workspace.terms[position]
        .term
        .try_reserve(term.len())
        .map_err(|_| AllocError::OutOfMemory)?;
    workspace.terms[position].term.push_str(term);
    if existing.is_none() {
        workspace.missing_terms += 1;
    }
    Ok(position)
}

fn prepare_terms(index: &InvertedIndex, workspace: &mut TextCommitWorkspace) -> RebindResult<()> {
    if workspace.documents.is_empty() {
        return Ok(());
    }
    for (term_number, (term, list)) in index.postings.iter().enumerate() {
        let mut changed_term = None;
        for (position, posting) in list.postings.iter().enumerate() {
            let Ok(document_index) = workspace
                .documents
                .binary_search_by_key(&posting.node_id, |document| document.id)
            else {
                continue;
            };
            if !committed_live(
                posting.created_epoch,
                posting.created_by,
                posting.deleted_epoch,
                posting.deleted_by,
                workspace.frontier,
                workspace.transaction_id,
            )? {
                continue;
            }
            let document = &mut workspace.documents[document_index];
            if document.old_len.is_none()
                || document.last_term == Some(term_number)
                || posting.term_freq == 0
            {
                return Err(DataRebindError::new(
                    "Text live posting has missing or duplicate document membership",
                ));
            }
            document.last_term = Some(term_number);
            document.posting_total = document
                .posting_total
                .checked_add(u64::from(posting.term_freq))
                .ok_or_else(|| DataRebindError::new("Text old document term count overflows"))?;
            let term_index = match changed_term {
                Some(term_index) => term_index,
                None => {
                    let term_index = add_term(workspace, term, Some(list))?;
                    changed_term = Some(term_index);
                    term_index
                }
            };
            reserve(&mut workspace.terms[term_index].close, 1)?;
            workspace.terms[term_index]
                .close
                .push((position, posting.clone()));
        }
    }
    for document in &workspace.documents {
        if document.posting_total != u64::from(document.old_len.map_or(0, |length| length)) {
            return Err(DataRebindError::new(
                "Text live document length differs from its postings",
            ));
        }
    }
    workspace
        .terms
        .sort_unstable_by(|left, right| left.term.cmp(&right.term));
    let old_terms = workspace.terms.len();
    let mut cursor = 0;
    while cursor < workspace.token_order.len() {
        let (document_index, token_index) = workspace.token_order[cursor];
        let term = &workspace.documents[document_index].tokens[token_index];
        let mut term_end = cursor + 1;
        while term_end < workspace.token_order.len() {
            let (other_doc, other_token) = workspace.token_order[term_end];
            if workspace.documents[other_doc].tokens[other_token] != *term {
                break;
            }
            term_end += 1;
        }
        let term_index = match workspace.terms[..old_terms]
            .binary_search_by(|change| change.term.as_str().cmp(term))
        {
            Ok(position) => position,
            Err(_) => {
                // Publish the empty builder to the workspace before allocating
                // its key. Split field borrows avoid any temporary key retiree.
                reserve(&mut workspace.terms, 1)?;
                let position = workspace.terms.len();
                let existing = index.postings.get(term);
                workspace.terms.push(TermChange {
                    term: String::new(),
                    existed: existing.is_some(),
                    expected_len: existing.map_or(0, |list| list.postings.len()),
                    insert_at: existing.map_or(0, |list| {
                        list.postings
                            .iter()
                            .position(|posting| posting.created_epoch == EpochId::PENDING)
                            .map_or(list.postings.len(), |position| position)
                    }),
                    close: Vec::new(),
                    additions: PostingList::default(),
                });
                workspace.terms[position]
                    .term
                    .try_reserve(term.len())
                    .map_err(|_| AllocError::OutOfMemory)?;
                workspace.terms[position].term.push_str(term);
                if existing.is_none() {
                    workspace.missing_terms += 1;
                }
                position
            }
        };
        while cursor < term_end {
            let document_index = workspace.token_order[cursor].0;
            let mut end = cursor + 1;
            while end < term_end && workspace.token_order[end].0 == document_index {
                end += 1;
            }
            let frequency = u32::try_from(end - cursor)
                .map_err(|_| DataRebindError::new("Text term frequency exceeds u32"))?;
            reserve(&mut workspace.terms[term_index].additions.postings, 1)?;
            workspace.terms[term_index]
                .additions
                .postings
                .push(VersionedPosting::new(
                    workspace.documents[document_index].id,
                    frequency,
                    workspace.commit_epoch,
                    None,
                ));
            cursor = end;
        }
    }
    Ok(())
}

fn prepare_aggregate(
    index: &InvertedIndex,
    workspace: &mut TextCommitWorkspace,
) -> RebindResult<()> {
    workspace.aggregate_len = index.agg_log.len();
    workspace.aggregate_insert_at = index
        .agg_log
        .iter()
        .position(|delta| delta.tx.is_some() || delta.epoch == EpochId::PENDING)
        .map_or(index.agg_log.len(), |position| position);
    let mut total = 0i64;
    let mut count = 0i64;
    for delta in &index.agg_log {
        if delta.tx == Some(workspace.transaction_id) {
            return Err(DataRebindError::new(
                "Text commit rejects own write-through pending aggregate",
            ));
        }
        if delta.tx.is_some() {
            continue;
        }
        if delta.epoch > workspace.frontier {
            return Err(DataRebindError::Conflict(
                "Text aggregate is newer than publication frontier",
            ));
        }
        total = total
            .checked_add(delta.d_total_len)
            .ok_or_else(|| DataRebindError::new("Text aggregate length overflows"))?;
        count = count
            .checked_add(delta.d_doc_count)
            .ok_or_else(|| DataRebindError::new("Text aggregate count overflows"))?;
    }
    let mut d_total_len = 0i64;
    let mut d_doc_count = 0i64;
    for document in &workspace.documents {
        let length_delta =
            i64::from(document.new_len) - i64::from(document.old_len.map_or(0, |length| length));
        let count_delta = i64::from(document.new_len != 0) - i64::from(document.old_len.is_some());
        d_total_len = d_total_len
            .checked_add(length_delta)
            .ok_or_else(|| DataRebindError::new("Text commit aggregate length overflows"))?;
        d_doc_count = d_doc_count
            .checked_add(count_delta)
            .ok_or_else(|| DataRebindError::new("Text commit aggregate count overflows"))?;
    }
    if total < 0
        || count < 0
        || total.checked_add(d_total_len).is_none_or(|value| value < 0)
        || count.checked_add(d_doc_count).is_none_or(|value| value < 0)
    {
        return Err(DataRebindError::new(
            "Text final aggregate is not representable",
        ));
    }
    workspace.aggregate_total = total;
    workspace.aggregate_count = count;
    if d_total_len != 0 || d_doc_count != 0 {
        reserve(&mut workspace.aggregate, 1)?;
        workspace.aggregate.push(AggDelta {
            epoch: workspace.commit_epoch,
            tx: None,
            d_total_len,
            d_doc_count,
        });
    }
    Ok(())
}

fn qualify(
    index: &InvertedIndex,
    workspace: &TextCommitWorkspace,
    scope: &TextCommitScope,
) -> RebindResult<()> {
    if !workspace.prepared
        || !scope.matches(index)
        || !workspace
            .gate
            .as_ref()
            .is_some_and(|gate| Arc::ptr_eq(gate, &index.mutation_scope_gate))
    {
        return Err(DataRebindError::new(
            "Text commit has no matching prepared scope",
        ));
    }
    if index
        .postings
        .capacity()
        .saturating_sub(index.postings.len())
        < workspace.missing_terms
        || index
            .doc_lengths
            .capacity()
            .saturating_sub(index.doc_lengths.len())
            < workspace.missing_documents
        || index.agg_log.len() != workspace.aggregate_len
        || index.agg_log.capacity().saturating_sub(index.agg_log.len()) < workspace.aggregate.len()
        || workspace.aggregate_insert_at > index.agg_log.len()
    {
        return Err(DataRebindError::new(
            "Text commit reserved map or aggregate capacity changed",
        ));
    }
    for term in &workspace.terms {
        match index.postings.get(&term.term) {
            Some(list)
                if term.existed
                    && list.postings.len() == term.expected_len
                    && list.postings.capacity().saturating_sub(list.postings.len())
                        >= term.additions.postings.len()
                    && term.insert_at <= list.postings.len()
                    && term
                        .close
                        .iter()
                        .all(|(position, _)| *position < list.postings.len()) => {}
            None if !term.existed => {}
            _ => return Err(DataRebindError::new("Text prepared posting target changed")),
        }
    }
    for document in &workspace.documents {
        match index.doc_lengths.get(&document.id) {
            Some(history)
                if document.existed
                    && history.len() == document.expected_len
                    && history.capacity().saturating_sub(history.len())
                        >= document.additions.len()
                    && document.insert_at <= history.len()
                    && document
                        .close
                        .as_ref()
                        .is_none_or(|(position, _)| *position < history.len()) => {}
            None if !document.existed => {}
            _ => {
                return Err(DataRebindError::new(
                    "Text prepared document target changed",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    static FAIL_RESERVATION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
mod tests;
