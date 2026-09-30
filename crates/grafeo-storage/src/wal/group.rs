//! Ordered commit-group seals carried by the current WAL envelope.

use super::WalEntry;
use blake3::hazmat::HasherExt;
use grafeo_common::utils::error::{Error, Result, StorageError};
use std::collections::HashMap;
use std::sync::OnceLock;

pub(super) const HEADER: &[u8; 10] = b"GRAFOWAL\x05\x00";
pub(super) const PREFIX_LEN: usize = HEADER.len() + 9;
pub(super) const SEAL_LEN: usize = 40;
const UNTAGGED: u64 = u64::MAX;
const MAX_PENDING_GROUPS: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Role {
    Data = 0,
    Commit = 1,
    Abort = 2,
    Independent = 3,
    Checkpoint = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Coordinate {
    role: Role,
    transaction: u64,
}

impl Coordinate {
    #[cfg(test)]
    pub(super) const fn untagged_data() -> Self {
        Self {
            role: Role::Data,
            transaction: UNTAGGED,
        }
    }
    pub(super) fn of<R: WalEntry>(record: &R) -> Result<Self> {
        let role = if record.is_commit() {
            Role::Commit
        } else if record.is_abort() {
            Role::Abort
        } else if record.is_checkpoint() {
            Role::Checkpoint
        } else if record.is_metadata() {
            Role::Independent
        } else {
            Role::Data
        };
        let transaction = match record.transaction_id() {
            Some(id) if id.is_valid() => id.as_u64(),
            Some(_) => return Err(Error::InvalidValue("invalid WAL group transaction".into())),
            None => UNTAGGED,
        };
        Ok(Self { role, transaction })
    }

    pub(super) fn bytes(self) -> [u8; 9] {
        let mut bytes = [0; 9];
        bytes[0] = self.role as u8;
        bytes[1..].copy_from_slice(&self.transaction.to_le_bytes());
        bytes
    }

    pub(super) fn has_seal(self) -> bool {
        matches!(self.role, Role::Commit | Role::Abort)
    }

    pub(super) fn is_checkpoint(self) -> bool {
        self.role == Role::Checkpoint
    }
}

pub(super) struct Envelope<'a> {
    pub(super) coordinate: Coordinate,
    pub(super) record: &'a [u8],
    unsigned: &'a [u8],
    seal: &'a [u8],
}

pub(super) fn envelope(bytes: &[u8]) -> std::result::Result<Envelope<'_>, String> {
    let Some((prefix, body)) = bytes.split_at_checked(PREFIX_LEN) else {
        return Err("missing or truncated current WAL group envelope".into());
    };
    if &prefix[..HEADER.len()] != HEADER {
        return Err("unsupported WAL record envelope or generation".into());
    }
    let role = match prefix[HEADER.len()] {
        0 => Role::Data,
        1 => Role::Commit,
        2 => Role::Abort,
        3 => Role::Independent,
        4 => Role::Checkpoint,
        _ => return Err("unknown WAL group role".into()),
    };
    let mut transaction = [0; 8];
    transaction.copy_from_slice(&prefix[HEADER.len() + 1..]);
    let coordinate = Coordinate {
        role,
        transaction: u64::from_le_bytes(transaction),
    };
    let seal_len = if coordinate.has_seal() { SEAL_LEN } else { 0 };
    let record_len = body
        .len()
        .checked_sub(seal_len)
        .ok_or_else(|| "truncated WAL group seal".to_string())?;
    if record_len == 0 {
        return Err("empty WAL record body".into());
    }
    let (record, seal) = body.split_at(record_len);
    Ok(Envelope {
        coordinate,
        record,
        unsigned: &bytes[..PREFIX_LEN + record_len],
        seal,
    })
}

struct Digest {
    count: u64,
    bytes: u64,
    hash: blake3::Hasher,
    active_writer: bool,
}

impl Default for Digest {
    fn default() -> Self {
        static CONTEXT: OnceLock<blake3::hazmat::ContextKey> = OnceLock::new();
        Self {
            count: 0,
            bytes: 0,
            hash: blake3::Hasher::new_from_context_key(CONTEXT.get_or_init(|| {
                blake3::hazmat::hash_derive_key_context("grafeo/wal/group-record/v5")
            })),
            active_writer: false,
        }
    }
}

#[derive(Default)]
pub(super) struct Groups {
    pending: HashMap<u64, Digest>,
    safe_segments: Vec<u64>,
    next_position: u64,
}

enum Update {
    Data {
        transaction: u64,
        count: u64,
        bytes: u64,
        length: u64,
        position: u64,
        next_position: u64,
        active_writer: bool,
    },
    Finish(u64),
    Checkpoint,
    Independent,
}

pub(super) struct Prepared {
    pub(super) seal: Option<[u8; SEAL_LEN]>,
    update: Update,
}

impl Groups {
    pub(super) fn reserve_segment(&mut self) -> Result<()> {
        self.safe_segments
            .try_reserve(1)
            .map_err(|error| Error::Io(std::io::Error::other(error)))
    }

    // Writers reserve before physical I/O; this publication cannot allocate.
    pub(super) fn record_segment_reserved(&mut self, sequence: u64) {
        if self.pending.is_empty()
            && self
                .safe_segments
                .last()
                .is_none_or(|previous| *previous < sequence)
        {
            self.safe_segments.push(sequence);
        }
    }

    pub(super) fn observe_segment(&mut self, sequence: u64) -> Result<()> {
        self.reserve_segment()?;
        self.record_segment_reserved(sequence);
        Ok(())
    }

    pub(super) fn initialize_empty_directory(&mut self, sequence: u64) -> Result<()> {
        if self.safe_segments.is_empty() {
            self.observe_segment(sequence)?;
        }
        Ok(())
    }

    pub(super) fn retirement_floor(&self, requested: u64) -> u64 {
        let end = self
            .safe_segments
            .partition_point(|sequence| *sequence <= requested);
        end.checked_sub(1)
            .and_then(|index| self.safe_segments.get(index))
            .or_else(|| self.safe_segments.first())
            .copied()
            .unwrap_or(0)
    }

    pub(super) fn retire_boundaries(&mut self, floor: u64) {
        self.safe_segments.retain(|sequence| *sequence >= floor);
    }

    pub(super) fn resume_writer(mut self) -> Self {
        // Recovered orphaned groups may be completed explicitly or discarded by
        // a checkpoint. New live writes must finish before checkpoint retirement.
        for digest in self.pending.values_mut() {
            digest.active_writer = false;
        }
        self
    }

    fn finish_seal(&self, frame: &Envelope<'_>) -> Result<[u8; SEAL_LEN]> {
        static CONTEXT: OnceLock<blake3::hazmat::ContextKey> = OnceLock::new();
        let mut hash =
            blake3::Hasher::new_from_context_key(CONTEXT.get_or_init(|| {
                blake3::hazmat::hash_derive_key_context("grafeo/wal/commit-group/v5")
            }));
        hash.update(frame.unsigned);
        let mut count = 0_u64;
        let mut append = |transaction: u64, digest: &Digest| -> Result<()> {
            count = count
                .checked_add(digest.count)
                .ok_or(Error::Storage(StorageError::Full))?;
            hash.update(&transaction.to_le_bytes());
            hash.update(&digest.count.to_le_bytes());
            hash.update(digest.hash.finalize().as_bytes());
            Ok(())
        };
        if frame.coordinate.transaction == UNTAGGED {
            // Untagged custom commit markers consume all pending records, as
            // generic recovery does. Sort coordinates, never hash table order.
            let mut keys = Vec::new();
            keys.try_reserve_exact(self.pending.len())
                .map_err(|error| Error::Io(std::io::Error::other(error)))?;
            keys.extend(self.pending.keys().copied());
            keys.sort_unstable();
            for key in keys {
                if let Some(digest) = self.pending.get(&key) {
                    append(key, digest)?;
                }
            }
        } else {
            for key in [frame.coordinate.transaction, UNTAGGED] {
                if let Some(digest) = self.pending.get(&key) {
                    append(key, digest)?;
                }
            }
        }
        let mut seal = [0; SEAL_LEN];
        seal[..8].copy_from_slice(&count.to_le_bytes());
        seal[8..].copy_from_slice(hash.finalize().as_bytes());
        Ok(seal)
    }

    pub(super) fn prepare(&mut self, bytes: &[u8], writing: bool) -> Result<Prepared> {
        let frame = envelope(bytes).map_err(Error::InvalidValue)?;
        self.prepare_envelope(&frame, writing)
    }

    pub(super) fn prepare_envelope(
        &mut self,
        frame: &Envelope<'_>,
        writing: bool,
    ) -> Result<Prepared> {
        let mut seal = None;
        let update = match frame.coordinate.role {
            Role::Data => {
                let prior = self.pending.get(&frame.coordinate.transaction);
                let is_new = prior.is_none();
                let count = prior
                    .map_or(0, |digest| digest.count)
                    .checked_add(1)
                    .ok_or(Error::Storage(StorageError::Full))?;
                // Length framing binds record boundaries as well as physical
                // order. Finalize once at the marker, not after every record.
                let length = u64::try_from(frame.unsigned.len())
                    .map_err(|_| Error::Storage(StorageError::Full))?;
                let position = self.next_position;
                let next_position = position
                    .checked_add(1)
                    .ok_or(Error::Storage(StorageError::Full))?;
                let bytes = prior
                    .map_or(0, |digest| digest.bytes)
                    .checked_add(16)
                    .and_then(|bytes| bytes.checked_add(length))
                    .ok_or(Error::Storage(StorageError::Full))?;
                let active_writer = writing || prior.is_some_and(|digest| digest.active_writer);
                if is_new {
                    if self.pending.len() == MAX_PENDING_GROUPS {
                        return Err(Error::InvalidValue(
                            "too many pending WAL transaction groups".into(),
                        ));
                    }
                    self.pending
                        .try_reserve(1)
                        .map_err(|error| Error::Io(std::io::Error::other(error)))?;
                }
                Update::Data {
                    transaction: frame.coordinate.transaction,
                    count,
                    bytes,
                    length,
                    position,
                    next_position,
                    active_writer,
                }
            }
            Role::Commit | Role::Abort => {
                if writing && frame.seal.iter().any(|byte| *byte != 0) {
                    return Err(Error::InvalidValue(
                        "WAL commit payload is already sealed".into(),
                    ));
                }
                let expected = self.finish_seal(frame)?;
                if !writing && frame.seal != expected {
                    return Err(Error::Storage(StorageError::InvalidWalEntry(
                        "WAL committed group count or digest mismatch".into(),
                    )));
                }
                seal = Some(expected);
                Update::Finish(frame.coordinate.transaction)
            }
            Role::Checkpoint => {
                if writing && self.pending.values().any(|digest| digest.active_writer) {
                    return Err(Error::InvalidValue(
                        "WAL checkpoint has unfinished live transaction groups".into(),
                    ));
                }
                Update::Checkpoint
            }
            Role::Independent => Update::Independent,
        };
        Ok(Prepared { seal, update })
    }

    pub(super) fn apply(&mut self, prepared: Prepared, payload: &[u8]) {
        match prepared.update {
            Update::Data {
                transaction,
                count,
                bytes,
                length,
                position,
                next_position,
                active_writer,
            } => {
                // Admission has reserved map capacity and checked the complete
                // stream byte count. This bounded update cannot allocate or
                // overflow BLAKE3's input counter, and only runs after success.
                let next = self.pending.entry(transaction).or_default();
                // A marker can consume tagged and untagged records together.
                // Bind their relative physical order across map coordinates.
                next.hash.update(&position.to_le_bytes());
                next.hash.update(&length.to_le_bytes());
                next.hash.update(payload);
                next.count = count;
                next.bytes = bytes;
                next.active_writer = active_writer;
                self.next_position = next_position;
            }
            Update::Finish(UNTAGGED) | Update::Checkpoint => self.pending.clear(),
            Update::Finish(transaction) => {
                self.pending.remove(&transaction);
                self.pending.remove(&UNTAGGED);
            }
            Update::Independent => {}
        }
        // Retirement only starts at an empty-group segment boundary, so a
        // reopened retained suffix reconstructs the same position sequence.
        if self.pending.is_empty() {
            self.next_position = 0;
        }
    }
}
