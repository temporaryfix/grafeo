//! Current, bounded Text WAL inputs for the existing sparse commit installer.

use super::super::{
    AggDelta, ExactAggDelta, ExactDocHistory, ExactDocLength, ExactPosting, ExactPostingList,
    ExactTextIndexImage, ExactTokenizerDescriptor, InvertedIndex, VersionedDocLen,
    VersionedPosting,
};
use super::{
    DocumentChange, RebindResult, TextCommitWorkspace, add_term, prepare_aggregate,
    prepare_documents, prepare_terms, reserve,
};
use crate::graph::lpg::DataRebindError;
use grafeo_common::types::{EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::Write;

const LIMIT: usize = 16 * 1024 * 1024;
const BIRTH: [u8; 4] = *b"TXB1";
const SPARSE: [u8; 4] = *b"TXS1";

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Header {
    k1: u64,
    b: u64,
    tokenizer: ExactTokenizerDescriptor,
    retained_from: EpochId,
}

impl Header {
    pub(super) fn capture(index: &InvertedIndex) -> Option<Self> {
        index.tokenizer_descriptor.clone().map(|tokenizer| Self {
            k1: index.config.k1.to_bits(),
            b: index.config.b.to_bits(),
            tokenizer,
            retained_from: index.retained_from,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Document {
    id: NodeId,
    prefix: usize,
    close: Option<(usize, ExactDocLength)>,
    additions: Vec<ExactDocLength>,
}

#[derive(Serialize, Deserialize)]
struct Term {
    term: String,
    prefix: usize,
    close: Vec<(usize, ExactPosting)>,
    additions: Vec<ExactPosting>,
}

#[derive(Serialize, Deserialize)]
struct Sparse {
    header: Header,
    frontier: EpochId,
    commit_epoch: EpochId,
    transaction_id: TransactionId,
    documents: Vec<Document>,
    terms: Vec<Term>,
    aggregate_prefix: usize,
    aggregate_total: i64,
    aggregate_count: i64,
    aggregate: Vec<ExactAggDelta>,
}

fn exact_document(entry: &VersionedDocLen) -> ExactDocLength {
    ExactDocLength {
        len: entry.len,
        created_epoch: entry.created_epoch,
        created_by: entry.created_by,
        deleted_epoch: entry.deleted_epoch,
        deleted_by: entry.deleted_by,
    }
}

fn exact_posting(entry: &VersionedPosting) -> ExactPosting {
    ExactPosting {
        node_id: entry.node_id,
        term_freq: entry.term_freq,
        created_epoch: entry.created_epoch,
        created_by: entry.created_by,
        deleted_epoch: entry.deleted_epoch,
        deleted_by: entry.deleted_by,
    }
}

fn exact_aggregate(entry: &AggDelta) -> ExactAggDelta {
    ExactAggDelta {
        epoch: entry.epoch,
        tx: entry.tx,
        d_total_len: entry.d_total_len,
        d_doc_count: entry.d_doc_count,
    }
}

impl TextCommitWorkspace {
    /// Owns opaque recorded input until the usual scoped preparation boundary.
    pub(crate) fn from_recorded(
        payload: Vec<u8>,
        frontier: EpochId,
        commit_epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Self {
        let mut workspace = Self::new(Vec::new(), frontier, commit_epoch, transaction_id);
        workspace.recorded = Some(payload);
        workspace
    }

    /// Encodes only prepared touched histories, never a surviving whole index.
    pub(crate) fn encode_wal_postimage(&self) -> Result<Vec<u8>> {
        if !self.prepared {
            return Err(Error::Serialization(
                "Text WAL input is not prepared".into(),
            ));
        }
        let header = self.wal_header.clone().ok_or_else(|| {
            Error::Serialization("Text tokenizer has no exact WAL descriptor".into())
        })?;
        let mut terms: Vec<_> = self
            .terms
            .iter()
            .map(|term| Term {
                term: term.term.clone(),
                prefix: term.insert_at,
                close: term
                    .close
                    .iter()
                    .map(|(position, entry)| (*position, exact_posting(entry)))
                    .collect(),
                additions: term.additions.postings.iter().map(exact_posting).collect(),
            })
            .collect();
        terms.sort_unstable_by(|left, right| left.term.cmp(&right.term));
        let image = Sparse {
            header,
            frontier: self.frontier,
            commit_epoch: self.commit_epoch,
            transaction_id: self.transaction_id,
            documents: self
                .documents
                .iter()
                .map(|document| Document {
                    id: document.id,
                    prefix: document.insert_at,
                    close: document
                        .close
                        .as_ref()
                        .map(|(position, entry)| (*position, exact_document(entry))),
                    additions: document.additions.iter().map(exact_document).collect(),
                })
                .collect(),
            terms,
            aggregate_prefix: self.aggregate_insert_at,
            aggregate_total: self.aggregate_total,
            aggregate_count: self.aggregate_count,
            aggregate: self.aggregate.iter().map(exact_aggregate).collect(),
        };
        validate_sparse(&image).map_err(DataRebindError::into_error)?;
        encode(SPARSE, &image)
    }
}

impl InvertedIndex {
    /// Exact current committed birth; pending histories have no durable image.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported tokenizers, pending or invalid exact
    /// history, allocation failure, or a payload exceeding its recovery bounds.
    pub fn encode_wal_birth(&self) -> Result<Vec<u8>> {
        let image = self.exact_committed_image().map_err(Error::Serialization)?;
        encode(BIRTH, &image)
    }

    /// Decodes an index born at this commit, never an arbitrary history image.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or oversized payloads, invalid exact
    /// state, or history that is not a live initial-floor birth at this commit.
    pub fn decode_wal_birth(bytes: &[u8], commit_epoch: EpochId) -> Result<Self> {
        let image: ExactTextIndexImage = decode(BIRTH, bytes)?;
        if commit_epoch == EpochId::INITIAL
            || commit_epoch == EpochId::PENDING
            || image.retained_from != EpochId::INITIAL
            || image.doc_lengths.iter().any(|document| {
                document.history.iter().any(|entry| {
                    !valid_creation(
                        entry.created_epoch,
                        entry.created_by,
                        entry.deleted_epoch,
                        entry.deleted_by,
                        commit_epoch,
                    )
                })
            })
            || image.postings.iter().any(|term| {
                term.postings.iter().any(|entry| {
                    !valid_creation(
                        entry.created_epoch,
                        entry.created_by,
                        entry.deleted_epoch,
                        entry.deleted_by,
                        commit_epoch,
                    )
                })
            })
            || image
                .agg_log
                .iter()
                .any(|entry| entry.epoch != commit_epoch || entry.tx.is_some())
        {
            return Err(Error::Serialization(
                "Text WAL birth is not an initial-floor live image at its commit epoch".into(),
            ));
        }
        Self::from_exact_wal_image(image)
    }

    fn from_exact_wal_image(image: ExactTextIndexImage) -> Result<Self> {
        let prepared = Self::prepare_exact_image(image).map_err(Error::Serialization)?;
        let mut index = Self::new(prepared.config.clone());
        let mutation = index.pin_prepared_restore().ok_or_else(|| {
            Error::Serialization("unpublished Text birth denied mutation authority".into())
        })?;
        index.install_prepared_image(prepared, &mutation);
        Ok(index)
    }
}

fn invalid() -> DataRebindError {
    DataRebindError::new("recorded Text fragment has invalid or mismatching exact state")
}

fn valid_creation(
    created: EpochId,
    creator: Option<TransactionId>,
    deleted: Option<EpochId>,
    deleter: Option<TransactionId>,
    expected_epoch: EpochId,
) -> bool {
    created == expected_epoch && creator.is_none() && deleted.is_none() && deleter.is_none()
}

fn validate_sparse(image: &Sparse) -> RebindResult<()> {
    if image.frontier == EpochId::PENDING
        || image.commit_epoch == EpochId::PENDING
        || image.commit_epoch <= image.frontier
        || image.transaction_id == TransactionId::INVALID
        || image.header.retained_from > image.frontier
        || image
            .documents
            .windows(2)
            .any(|pair| matches!(pair, [left, right] if left.id >= right.id))
        || image
            .terms
            .windows(2)
            .any(|pair| matches!(pair, [left, right] if left.term >= right.term))
    {
        return Err(invalid());
    }
    for document in &image.documents {
        if !document.id.is_valid() || document.additions.len() > 1 {
            return Err(invalid());
        }
        if let Some((position, entry)) = &document.close
            && (*position >= document.prefix
                || entry.len == 0
                || entry.created_epoch > image.frontier
                || !valid_creation(
                    entry.created_epoch,
                    entry.created_by,
                    entry.deleted_epoch,
                    entry.deleted_by,
                    entry.created_epoch,
                ))
        {
            return Err(invalid());
        }
        for entry in &document.additions {
            if entry.len == 0
                || !valid_creation(
                    entry.created_epoch,
                    entry.created_by,
                    entry.deleted_epoch,
                    entry.deleted_by,
                    image.commit_epoch,
                )
            {
                return Err(invalid());
            }
        }
    }
    for term in &image.terms {
        if term.term.is_empty()
            || (term.close.is_empty() && term.additions.is_empty())
            || term
                .close
                .windows(2)
                .any(|pair| matches!(pair, [left, right] if left.0 >= right.0))
            || term
                .additions
                .windows(2)
                .any(|pair| matches!(pair, [left, right] if left.node_id >= right.node_id))
        {
            return Err(invalid());
        }
        for (position, entry) in &term.close {
            if *position >= term.prefix
                || !entry.node_id.is_valid()
                || entry.term_freq == 0
                || entry.created_epoch > image.frontier
                || !valid_creation(
                    entry.created_epoch,
                    entry.created_by,
                    entry.deleted_epoch,
                    entry.deleted_by,
                    entry.created_epoch,
                )
            {
                return Err(invalid());
            }
        }
        for entry in &term.additions {
            if !entry.node_id.is_valid()
                || entry.term_freq == 0
                || !valid_creation(
                    entry.created_epoch,
                    entry.created_by,
                    entry.deleted_epoch,
                    entry.deleted_by,
                    image.commit_epoch,
                )
            {
                return Err(invalid());
            }
        }
    }
    if image.aggregate.len() > 1
        || image
            .aggregate
            .iter()
            .any(|entry| entry.epoch != image.commit_epoch || entry.tx.is_some())
    {
        return Err(invalid());
    }
    Ok(())
}

pub(super) fn prepare_recorded(
    index: &InvertedIndex,
    workspace: &mut TextCommitWorkspace,
    payload: &[u8],
) -> RebindResult<()> {
    let image: Sparse = decode(SPARSE, payload).map_err(|_| invalid())?;
    validate_sparse(&image)?;
    if image.frontier != workspace.frontier
        || image.commit_epoch != workspace.commit_epoch
        || image.transaction_id != workspace.transaction_id
        || Header::capture(index).as_ref() != Some(&image.header)
    {
        return Err(invalid());
    }
    workspace.wal_header = Some(image.header);
    reserve(&mut workspace.documents, image.documents.len())?;
    for document in &image.documents {
        let mut additions = Vec::new();
        reserve(&mut additions, document.additions.len())?;
        additions.extend(
            document
                .additions
                .iter()
                .map(|entry| VersionedDocLen::new(entry.len, entry.created_epoch, None)),
        );
        workspace.documents.push(DocumentChange {
            id: document.id,
            input: 0,
            tokens: Vec::new(),
            new_len: document.additions.first().map_or(0, |entry| entry.len),
            old_len: None,
            expected_len: 0,
            existed: false,
            close: None,
            insert_at: 0,
            additions,
            posting_total: 0,
            last_term: None,
        });
    }
    // This inspects touched preimages only. No text/tokenizer input is supplied.
    prepare_documents(index, workspace)?;
    for (actual, recorded) in workspace.documents.iter().zip(&image.documents) {
        if actual.insert_at != recorded.prefix
            || actual
                .close
                .as_ref()
                .map(|(position, entry)| (*position, exact_document(entry)))
                != recorded.close
        {
            return Err(invalid());
        }
    }
    // Discover every live posting of touched documents, including omitted-close
    // attempts. The same discovery is required by ordinary sparse preparation.
    prepare_terms(index, workspace)?;
    let old_terms = workspace.terms.len();
    for actual in &workspace.terms {
        if image
            .terms
            .binary_search_by(|term| term.term.cmp(&actual.term))
            .is_err()
        {
            return Err(invalid());
        }
    }
    for recorded in image.terms {
        let position = match workspace
            .terms
            .get(..old_terms)
            .ok_or_else(invalid)?
            .binary_search_by(|term| term.term.cmp(&recorded.term))
        {
            Ok(position) => position,
            Err(_) => add_term(
                workspace,
                &recorded.term,
                index.postings.get(&recorded.term),
            )?,
        };
        let actual = workspace.terms.get_mut(position).ok_or_else(invalid)?;
        if actual.insert_at != recorded.prefix
            || actual.close.len() != recorded.close.len()
            || actual.close.iter().zip(&recorded.close).any(
                |((position, entry), (expected_position, expected))| {
                    position != expected_position || exact_posting(entry) != *expected
                },
            )
        {
            return Err(invalid());
        }
        reserve(&mut actual.additions.postings, recorded.additions.len())?;
        actual
            .additions
            .postings
            .extend(recorded.additions.into_iter().map(|entry| {
                VersionedPosting::new(entry.node_id, entry.term_freq, entry.created_epoch, None)
            }));
    }
    for document in &mut workspace.documents {
        document.posting_total = 0;
    }
    for term in &workspace.terms {
        for posting in &term.additions.postings {
            let position = workspace
                .documents
                .binary_search_by_key(&posting.node_id, |document| document.id)
                .map_err(|_| invalid())?;
            let document = workspace.documents.get_mut(position).ok_or_else(invalid)?;
            document.posting_total = document
                .posting_total
                .checked_add(u64::from(posting.term_freq))
                .ok_or_else(invalid)?;
        }
    }
    if workspace
        .documents
        .iter()
        .any(|document| document.posting_total != u64::from(document.new_len))
    {
        return Err(invalid());
    }
    prepare_aggregate(index, workspace)?;
    if workspace.aggregate_insert_at != image.aggregate_prefix
        || workspace.aggregate_total != image.aggregate_total
        || workspace.aggregate_count != image.aggregate_count
        || workspace.aggregate.len() != image.aggregate.len()
        || workspace
            .aggregate
            .iter()
            .zip(&image.aggregate)
            .any(|(actual, recorded)| exact_aggregate(actual) != *recorded)
    {
        return Err(invalid());
    }
    Ok(())
}

struct Bounded(Vec<u8>);

impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > LIMIT.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("Text WAL exceeds 16 MiB"));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(std::io::Error::other)?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode<T: Serialize + DeserializeOwned>(magic: [u8; 4], image: &T) -> Result<Vec<u8>> {
    let mut output = Bounded(Vec::new());
    output
        .write_all(&magic)
        .map_err(|error| Error::Serialization(error.to_string()))?;
    bincode::serde::encode_into_std_write(image, &mut output, bincode::config::standard())
        .map_err(|error| Error::Serialization(error.to_string()))?;
    // Bincode's decode accounting includes expanded integers and allocations;
    // a byte-sized limit alone would admit records recovery cannot decode.
    let _: T = decode(magic, &output.0)?;
    Ok(output.0)
}

fn decode<T: DeserializeOwned>(magic: [u8; 4], bytes: &[u8]) -> Result<T> {
    if bytes.len() > LIMIT || !bytes.starts_with(&magic) {
        return Err(Error::Serialization(
            "unsupported or oversized Text WAL payload".into(),
        ));
    }
    let body = bytes
        .get(magic.len()..)
        .ok_or_else(|| Error::Serialization("truncated Text WAL payload".into()))?;
    preflight(magic, body)
        .map_err(|()| Error::Serialization("malformed or oversized Text WAL allocation".into()))?;
    let (image, consumed) =
        bincode::serde::decode_from_slice(body, bincode::config::standard().with_limit::<LIMIT>())
            .map_err(|error| Error::Serialization(error.to_string()))?;
    if consumed != body.len() {
        return Err(Error::Serialization(
            "Text WAL payload has trailing bytes".into(),
        ));
    }
    Ok(image)
}

// No allocation occurs before this scan has bounded all declared collections
// and strings, including birth's existing exact-image serde representation.
struct Preflight<'a> {
    bytes: &'a [u8],
    position: usize,
    heap: usize,
}

impl Preflight<'_> {
    fn take(&mut self, count: usize) -> std::result::Result<&[u8], ()> {
        let end = self.position.checked_add(count).ok_or(())?;
        let bytes = self.bytes.get(self.position..end).ok_or(())?;
        self.position = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> std::result::Result<u8, ()> {
        self.take(1)?.first().copied().ok_or(())
    }
    fn int(&mut self) -> std::result::Result<u64, ()> {
        let (value, minimum) = match self.byte()? {
            value @ 0..=250 => return Ok(u64::from(value)),
            251 => (
                u64::from(u16::from_le_bytes(
                    self.take(2)?.try_into().map_err(|_| ())?,
                )),
                251,
            ),
            252 => (
                u64::from(u32::from_le_bytes(
                    self.take(4)?.try_into().map_err(|_| ())?,
                )),
                65_536,
            ),
            253 => (
                u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| ())?),
                4_294_967_296,
            ),
            _ => return Err(()),
        };
        if value < minimum {
            return Err(());
        }
        Ok(value)
    }
    fn option(
        &mut self,
        value: impl FnOnce(&mut Self) -> std::result::Result<(), ()>,
    ) -> std::result::Result<(), ()> {
        match self.byte()? {
            0 => Ok(()),
            1 => value(self),
            _ => Err(()),
        }
    }
    fn optional_int(&mut self) -> std::result::Result<(), ()> {
        self.option(|wire| {
            wire.int()?;
            Ok(())
        })
    }
    fn sequence<T>(&mut self) -> std::result::Result<usize, ()> {
        let count = usize::try_from(self.int()?).map_err(|_| ())?;
        self.heap = self
            .heap
            .checked_add(count.checked_mul(std::mem::size_of::<T>()).ok_or(())?)
            .ok_or(())?;
        if self.heap > LIMIT || count > self.bytes.len().checked_sub(self.position).ok_or(())? {
            return Err(());
        }
        Ok(count)
    }
    fn string(&mut self) -> std::result::Result<(), ()> {
        let length = self.sequence::<u8>()?;
        std::str::from_utf8(self.take(length)?).map_err(|_| ())?;
        Ok(())
    }
    fn configuration(&mut self, bits: bool) -> std::result::Result<(), ()> {
        if bits {
            self.int()?;
            self.int()?;
        } else {
            self.take(16)?;
        }
        if self.int()? != 0 {
            return Err(());
        }
        usize::try_from(self.int()?).map_err(|_| ())?;
        if self.int()? == EpochId::PENDING.as_u64() {
            return Err(());
        }
        Ok(())
    }
    fn document(&mut self) -> std::result::Result<(), ()> {
        self.int()?;
        self.int()?;
        self.optional_int()?;
        self.optional_int()?;
        self.optional_int()
    }
    fn posting(&mut self) -> std::result::Result<(), ()> {
        self.int()?;
        self.document()
    }
    fn aggregate(&mut self) -> std::result::Result<(), ()> {
        for _ in 0..self.sequence::<ExactAggDelta>()? {
            self.int()?;
            self.optional_int()?;
            self.int()?;
            self.int()?;
        }
        Ok(())
    }
}

fn preflight(magic: [u8; 4], bytes: &[u8]) -> std::result::Result<(), ()> {
    let mut wire = Preflight {
        bytes,
        position: 0,
        heap: 0,
    };
    if magic == BIRTH {
        wire.configuration(false)?;
        for _ in 0..wire.sequence::<ExactPostingList>()? {
            wire.string()?;
            for _ in 0..wire.sequence::<ExactPosting>()? {
                wire.posting()?;
            }
        }
        for _ in 0..wire.sequence::<ExactDocHistory>()? {
            wire.int()?;
            for _ in 0..wire.sequence::<ExactDocLength>()? {
                wire.document()?;
            }
        }
        wire.aggregate()?;
    } else {
        wire.configuration(true)?;
        wire.int()?;
        wire.int()?;
        wire.int()?;
        for _ in 0..wire.sequence::<Document>()? {
            wire.int()?;
            wire.int()?;
            wire.option(|wire| {
                wire.int()?;
                wire.document()
            })?;
            for _ in 0..wire.sequence::<ExactDocLength>()? {
                wire.document()?;
            }
        }
        for _ in 0..wire.sequence::<Term>()? {
            wire.string()?;
            wire.int()?;
            for _ in 0..wire.sequence::<(usize, ExactPosting)>()? {
                wire.int()?;
                wire.posting()?;
            }
            for _ in 0..wire.sequence::<ExactPosting>()? {
                wire.posting()?;
            }
        }
        wire.int()?;
        wire.int()?;
        wire.int()?;
        wire.aggregate()?;
    }
    if wire.position != bytes.len() {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AggDelta, BIRTH, DataRebindError, Document, EpochId, Error, ExactAggDelta, InvertedIndex,
        LIMIT, NodeId, Result, SPARSE, Sparse, TextCommitWorkspace, TransactionId, VersionedDocLen,
        VersionedPosting, decode, encode,
    };
    use crate::index::text::BM25Config;

    const P: EpochId = EpochId::new(5);
    const C: EpochId = EpochId::new(6);
    const TX: TransactionId = TransactionId::new(42);

    // Historical baselines are legal exact images, but are not legal WAL births.
    fn decode_baseline(bytes: &[u8]) -> Result<InvertedIndex> {
        InvertedIndex::from_exact_wal_image(decode(BIRTH, bytes)?)
    }

    fn scores_at(index: &InvertedIndex, epoch: EpochId) -> Result<Vec<(NodeId, u64)>> {
        let mut scores: Vec<_> = index
            .search_visible(
                "shared",
                10,
                epoch,
                TransactionId::INVALID,
                &[],
                &Default::default(),
            )?
            .into_iter()
            .map(|(id, score)| (id, score.to_bits()))
            .collect();
        scores.sort_unstable_by_key(|(id, _)| *id);
        Ok(scores)
    }

    fn fixture() -> Result<InvertedIndex> {
        let mut index = InvertedIndex::with_simple_tokenizer(BM25Config { k1: 2.3, b: 0.4 }, 1);
        index.insert_versioned(NodeId::new(1), "first shared", EpochId::new(1), None);
        index.insert_versioned(NodeId::new(1), "old shared", EpochId::new(3), None);
        index.insert_versioned(NodeId::new(2), "delete shared", EpochId::new(3), None);
        index.insert_versioned(NodeId::new(3), "untouched", EpochId::new(1), None);
        index.gc(EpochId::new(2))?;
        Ok(index)
    }

    fn prepare_and_install(
        index: &mut InvertedIndex,
        changes: Vec<(NodeId, Option<String>)>,
    ) -> Result<Vec<u8>> {
        let mut workspace = TextCommitWorkspace::new(changes, P, C, TX);
        let scope = index.pin_commit_scope(&mut workspace)?;
        index.prepare_commit_fragments(&mut workspace, &scope)?;
        let bytes = workspace.encode_wal_postimage()?;
        workspace
            .validate_prepared(index, &scope)
            .map_err(DataRebindError::into_error)?;
        workspace.install_prequalified(index, &scope);
        assert!(
            workspace.encode_wal_postimage().is_err(),
            "consumed fragments cannot be recorded again"
        );
        Ok(bytes)
    }

    fn replay(index: &mut InvertedIndex, bytes: Vec<u8>) -> Result<()> {
        let mut workspace = TextCommitWorkspace::from_recorded(bytes, P, C, TX);
        let scope = index.pin_commit_scope(&mut workspace)?;
        index.prepare_commit_fragments(&mut workspace, &scope)?;
        workspace
            .validate_prepared(index, &scope)
            .map_err(DataRebindError::into_error)?;
        workspace.install_prequalified(index, &scope);
        Ok(())
    }

    #[test]
    fn recorded_sparse_matches_exact_retained_image_and_scores() -> Result<()> {
        let mut live = fixture()?;
        let mut recovered = decode_baseline(&live.encode_wal_birth()?)?;
        let payload = prepare_and_install(
            &mut live,
            vec![
                (NodeId::new(1), Some("new new shared".into())),
                (NodeId::new(2), None),
                (NodeId::new(4), Some("born".into())),
            ],
        )?;
        replay(&mut recovered, payload.clone())?;
        assert_eq!(live.encode_wal_birth()?, recovered.encode_wal_birth()?);
        for epoch in [EpochId::new(2), EpochId::new(3), P, C] {
            assert_eq!(scores_at(&live, epoch)?, scores_at(&recovered, epoch)?);
        }
        let before = recovered.encode_wal_birth()?;
        assert!(replay(&mut recovered, payload).is_err());
        assert_eq!(before, recovered.encode_wal_birth()?);
        Ok(())
    }

    #[test]
    fn recorded_prefix_excludes_foreign_pending_suffixes() -> Result<()> {
        let mut live = fixture()?;
        let mut recovered = decode_baseline(&live.encode_wal_birth()?)?;
        let foreign = TransactionId::new(99);
        live.insert_versioned(
            NodeId::new(9),
            "shared pending",
            EpochId::PENDING,
            Some(foreign),
        );
        live.doc_lengths
            .get_mut(&NodeId::new(1))
            .ok_or_else(|| Error::Serialization("fixture document missing".into()))?
            .push(VersionedDocLen::new(1, EpochId::PENDING, Some(foreign)));
        live.postings
            .get_mut("shared")
            .ok_or_else(|| Error::Serialization("fixture term missing".into()))?
            .postings
            .push(VersionedPosting::new(
                NodeId::new(1),
                1,
                EpochId::PENDING,
                Some(foreign),
            ));
        live.agg_log.push(AggDelta {
            epoch: EpochId::PENDING,
            tx: Some(foreign),
            d_total_len: 1,
            d_doc_count: 1,
        });
        let payload =
            prepare_and_install(&mut live, vec![(NodeId::new(1), Some("shared new".into()))])?;
        let image: Sparse = decode(SPARSE, &payload)?;
        assert_eq!(image.documents[0].prefix, 2);
        assert!(image.aggregate_prefix < live.agg_log.len());
        replay(&mut recovered, payload)?;
        assert_eq!(scores_at(&live, C)?, scores_at(&recovered, C)?);
        assert!(
            live.doc_lengths[&NodeId::new(1)]
                .iter()
                .any(|entry| entry.created_by == Some(foreign))
        );
        assert!(
            live.encode_wal_birth().is_err(),
            "birth cannot silently discard pending state"
        );
        // Remove only the test-injected uncommitted suffix; durable committed
        // state must then be byte-identical, not merely search-equivalent.
        live.doc_lengths.retain(|_, history| {
            history.retain(|entry| entry.created_by != Some(foreign));
            !history.is_empty()
        });
        live.postings.retain(|_, list| {
            list.postings
                .retain(|entry| entry.created_by != Some(foreign));
            !list.postings.is_empty()
        });
        live.agg_log.retain(|delta| delta.tx != Some(foreign));
        assert_eq!(live.encode_wal_birth()?, recovered.encode_wal_birth()?);
        Ok(())
    }

    #[test]
    fn recorded_wrong_floor_close_membership_aggregate_and_epochs_refuse_unchanged() -> Result<()> {
        let mut live = fixture()?;
        let birth = live.encode_wal_birth()?;
        let payload =
            prepare_and_install(&mut live, vec![(NodeId::new(1), Some("new shared".into()))])?;
        for corruption in 0..13 {
            let mut image: Sparse = decode(SPARSE, &payload)?;
            match corruption {
                0 => image.header.retained_from = EpochId::INITIAL,
                1 => image.documents[0].prefix += 1,
                2 => {
                    if let Some((_, entry)) = &mut image.documents[0].close {
                        entry.len += 1;
                    }
                }
                3 => image.terms.retain(|term| term.term != "old"),
                4 => {
                    if let Some(term) = image
                        .terms
                        .iter_mut()
                        .find(|term| !term.additions.is_empty())
                    {
                        term.additions[0].term_freq += 1;
                    }
                }
                5 => image.aggregate_total += 1,
                6 => image.commit_epoch = EpochId::PENDING,
                7 => {
                    if let Some(term) = image.terms.iter_mut().find(|term| !term.close.is_empty()) {
                        term.close[0].1.created_epoch = EpochId::new(2);
                    }
                }
                8 => image.aggregate_prefix += 1,
                9 => image.documents.push(Document {
                    id: image.documents[0].id,
                    prefix: 0,
                    close: None,
                    additions: Vec::new(),
                }),
                10 => image.documents[0].additions[0].created_by = Some(TransactionId::new(99)),
                11 => image.header.k1 = 2.4_f64.to_bits(),
                _ => image.aggregate.push(ExactAggDelta {
                    epoch: C,
                    tx: None,
                    d_total_len: 1,
                    d_doc_count: 0,
                }),
            }
            let mut recovered = decode_baseline(&birth)?;
            // Raw encoding lets malformed semantic images reach the real
            // preparation validator; production never accepts them unchecked.
            let bytes = encode(SPARSE, &image)?;
            assert!(
                replay(&mut recovered, bytes).is_err(),
                "corruption {corruption}"
            );
            assert_eq!(
                birth,
                recovered.encode_wal_birth()?,
                "corruption {corruption}"
            );
        }
        Ok(())
    }

    #[test]
    fn sparse_payload_tracks_touched_rows_not_whole_index() -> Result<()> {
        let mut sizes = Vec::new();
        for unrelated in [10, 2_000] {
            let mut index = fixture()?;
            for id in 10..10 + unrelated {
                index.insert_versioned(
                    NodeId::new(id),
                    &format!("unrelated{id}"),
                    EpochId::new(4),
                    None,
                );
            }
            let bytes = prepare_and_install(
                &mut index,
                vec![(NodeId::new(1), Some("new shared".into()))],
            )?;
            let image: Sparse = decode(SPARSE, &bytes)?;
            assert_eq!(image.documents.len(), 1);
            assert_eq!(image.terms.len(), 3);
            sizes.push(bytes.len());
        }
        assert!(
            sizes[1].abs_diff(sizes[0]) < 32,
            "only prefix counters may grow: {sizes:?}"
        );
        Ok(())
    }

    #[test]
    fn encoder_rejects_byte_sized_but_unreadable_allocation_budget() -> Result<()> {
        let mut index = InvertedIndex::with_simple_tokenizer(BM25Config::default(), 1);
        let bytes = prepare_and_install(&mut index, vec![(NodeId::new(1), Some("alpha".into()))])?;
        let mut image: Sparse = decode(SPARSE, &bytes)?;
        image.terms[0].term = "x".repeat(LIMIT - 128);
        let mut unchecked = SPARSE.to_vec();
        unchecked.extend(
            bincode::serde::encode_to_vec(&image, bincode::config::standard())
                .map_err(|error| Error::Serialization(error.to_string()))?,
        );
        assert!(unchecked.len() < LIMIT, "physical bytes alone fit");
        assert!(decode::<Sparse>(SPARSE, &unchecked).is_err());
        assert!(
            encode(SPARSE, &image).is_err(),
            "encoder must not admit self-unreadable input"
        );
        assert!(InvertedIndex::decode_wal_birth(b"TXB0", C).is_err());
        assert!(InvertedIndex::decode_wal_birth(&vec![0; LIMIT + 1], C).is_err());
        Ok(())
    }

    #[test]
    fn birth_requires_current_commit_live_rows_and_initial_retention_floor() -> Result<()> {
        for created in [P, C, EpochId::new(7)] {
            let mut index = InvertedIndex::with_simple_tokenizer(BM25Config::default(), 1);
            index.insert_versioned(NodeId::new(1), "alpha beta", created, None);
            let bytes = index.encode_wal_birth()?;
            assert!(
                decode_baseline(&bytes).is_ok(),
                "valid standalone image at {created}"
            );
            if created == C {
                assert_eq!(
                    bytes,
                    InvertedIndex::decode_wal_birth(&bytes, C)?.encode_wal_birth()?
                );
                assert!(InvertedIndex::decode_wal_birth(&bytes, EpochId::INITIAL).is_err());
                assert!(InvertedIndex::decode_wal_birth(&bytes, EpochId::PENDING).is_err());
                index.gc(C)?;
                let retained = index.encode_wal_birth()?;
                assert!(decode_baseline(&retained).is_ok());
                assert!(InvertedIndex::decode_wal_birth(&retained, C).is_err());
            } else {
                assert!(
                    InvertedIndex::decode_wal_birth(&bytes, C).is_err(),
                    "birth epoch {created} differs from C"
                );
            }
        }
        let mut closed = InvertedIndex::with_simple_tokenizer(BM25Config::default(), 1);
        closed.insert_versioned(NodeId::new(1), "alpha", C, None);
        assert!(closed.remove_versioned(NodeId::new(1), C, None));
        let bytes = closed.encode_wal_birth()?;
        assert!(decode_baseline(&bytes).is_ok());
        assert!(
            InvertedIndex::decode_wal_birth(&bytes, C).is_err(),
            "same-C closed history is not a birth"
        );
        let empty = InvertedIndex::new(BM25Config::default()).encode_wal_birth()?;
        assert!(InvertedIndex::decode_wal_birth(&empty, C).is_ok());
        Ok(())
    }
}
