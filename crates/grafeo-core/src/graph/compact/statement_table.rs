//! Statement / claim table: seven-column opaque claim layout as temporal rows.
//!
//! Hashes are mirrored **opaquely** ([`grafeo_common::types::ContentId`]); grafeo never computes or
//! interprets them. Transaction time is [`EpochInterval`]; `timestamp_ns` is
//! first-class valid/phenomenon time.

use grafeo_common::types::{ContentId, EpochId, EpochInterval, Interner, LocalId};

/// Refusal from the bounded, atomic statement-table insertion path.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StatementTableInsertError {
    /// Statement rows must be associated with a committed epoch.
    #[error("statement rows require a committed transaction epoch")]
    PendingEpoch,
    /// The table's dense v1 `u32` row-id space would be exhausted.
    #[error(
        "statement batch of {incoming} rows exceeds the v1 LocalId capacity from current length {current}"
    )]
    RowCapacityExceeded {
        /// Rows already present.
        current: usize,
        /// Rows requested by this batch.
        incoming: usize,
    },
    /// One assertion cannot be represented by the v1 `u32` length field.
    #[error(
        "statement assertion {batch_index} has {bytes} bytes and exceeds the v1 u32 length field"
    )]
    AssertionTooLarge {
        /// Zero-based row within the rejected batch.
        batch_index: usize,
        /// Assertion byte length.
        bytes: usize,
    },
}

/// One ingested statement in the seven-column claim layout.
///
/// All hash fields are opaque bytes supplied by the producer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatementIngest {
    /// Producer kind tag (opaque).
    pub target_kind: u8,
    /// Opaque 32-byte target identity supplied by the application.
    pub target_ref: [u8; 32],
    /// Producer row / offset (opaque to grafeo).
    pub target_row: u64,
    /// Producer meta kind (opaque).
    pub meta_kind: u8,
    /// Assertion payload (opaque bytes; not parsed).
    pub assertion: Vec<u8>,
    /// Phenomenon / valid time (nanoseconds). Queryable independently of
    /// transaction-time [`EpochInterval`].
    pub timestamp_ns: i64,
    /// Opaque 32-byte source identity.
    pub source_id: [u8; 32],
}

/// A stored statement row with assigned [`LocalId`] and transaction-time validity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatementRow {
    /// Dense per-store row id (not durable across rebuilds of the interner).
    pub id: LocalId,
    /// Producer kind tag.
    pub target_kind: u8,
    /// Opaque target content-address.
    pub target_ref: ContentId,
    /// Producer row / offset.
    pub target_row: u64,
    /// Producer meta kind.
    pub meta_kind: u8,
    /// Assertion payload.
    pub assertion: Vec<u8>,
    /// Phenomenon / valid time.
    pub timestamp_ns: i64,
    /// Opaque source content-address.
    pub source_id: ContentId,
    /// Transaction-time interval (`[from, to)`).
    pub validity: EpochInterval,
}

/// Append-only temporal statement table.
#[derive(Clone, Debug, Default)]
pub struct StatementTable {
    rows: Vec<StatementRow>,
    refs: Interner<ContentId>,
}

impl StatementTable {
    /// Empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored rows (all transaction-time lives).
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Appends `batch` as open rows at `tx_epoch`. Returns assigned row ids.
    ///
    /// Hashes are interned opaquely. Same bytes intern to the same
    /// [`ContentId`]; row ids are assigned in insertion order.
    ///
    /// # Panics
    ///
    /// Panics before mutating the table if `tx_epoch` is the uncommitted
    /// [`EpochId::PENDING`] sentinel, if the batch would exhaust valid
    /// [`LocalId`] values, or if an assertion cannot be represented by the v1
    /// wire format. The database facade validates these conditions and returns
    /// a structured error before calling this parked in-memory primitive.
    pub fn insert_batch(&mut self, batch: &[StatementIngest], tx_epoch: EpochId) -> Vec<LocalId> {
        self.try_insert_batch(batch, tx_epoch)
            .expect("statement-table insertion preconditions must hold")
    }

    /// Atomically appends `batch` after validating every v1 wire invariant.
    ///
    /// # Errors
    ///
    /// Returns an error without changing rows or interned references when the
    /// epoch is uncommitted, the dense row-id space would be exhausted, or an
    /// assertion is too large for the portable v1 format.
    pub fn try_insert_batch(
        &mut self,
        batch: &[StatementIngest],
        tx_epoch: EpochId,
    ) -> Result<Vec<LocalId>, StatementTableInsertError> {
        if tx_epoch == EpochId::PENDING {
            return Err(StatementTableInsertError::PendingEpoch);
        }
        let new_len = self.rows.len().checked_add(batch.len()).ok_or(
            StatementTableInsertError::RowCapacityExceeded {
                current: self.rows.len(),
                incoming: batch.len(),
            },
        )?;
        if u32::try_from(new_len).is_err() {
            return Err(StatementTableInsertError::RowCapacityExceeded {
                current: self.rows.len(),
                incoming: batch.len(),
            });
        }
        for (batch_index, item) in batch.iter().enumerate() {
            if u32::try_from(item.assertion.len()).is_err() {
                return Err(StatementTableInsertError::AssertionTooLarge {
                    batch_index,
                    bytes: item.assertion.len(),
                });
            }
        }

        let ids = (self.rows.len()..new_len)
            .map(|row| {
                u32::try_from(row).map(LocalId::new).map_err(|_| {
                    StatementTableInsertError::RowCapacityExceeded {
                        current: self.rows.len(),
                        incoming: batch.len(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (item, &id) in batch.iter().zip(&ids) {
            let target_ref = ContentId::from_bytes(item.target_ref);
            let source_id = ContentId::from_bytes(item.source_id);
            self.refs.intern(target_ref);
            self.refs.intern(source_id);
            self.rows.push(StatementRow {
                id,
                target_kind: item.target_kind,
                target_ref,
                target_row: item.target_row,
                meta_kind: item.meta_kind,
                assertion: item.assertion.clone(),
                timestamp_ns: item.timestamp_ns,
                source_id,
                validity: EpochInterval::open(tx_epoch),
            });
        }
        Ok(ids)
    }

    /// Rows whose transaction-time interval contains `epoch`.
    ///
    /// `PENDING` matches still-open rows (current view).
    #[must_use]
    pub fn statements_at_epoch(&self, epoch: EpochId) -> Vec<&StatementRow> {
        self.rows
            .iter()
            .filter(|row| {
                if epoch == EpochId::PENDING {
                    row.validity.is_open()
                } else {
                    row.validity.contains(epoch)
                }
            })
            .collect()
    }

    /// Rows whose phenomenon/valid-time `timestamp_ns` equals `timestamp_ns`.
    #[must_use]
    pub fn statements_at_valid_time(&self, timestamp_ns: i64) -> Vec<&StatementRow> {
        self.rows
            .iter()
            .filter(|row| row.timestamp_ns == timestamp_ns)
            .collect()
    }

    /// Rows whose opaque `target_ref` matches `target`.
    #[must_use]
    pub fn statements_for_target(&self, target: ContentId) -> Vec<&StatementRow> {
        self.rows
            .iter()
            .filter(|row| row.target_ref == target)
            .collect()
    }

    /// Deterministic serialization (insertion order).
    ///
    /// # Panics
    ///
    /// Panics only if internal invariants were bypassed. All public insertion
    /// and decoding paths bound row and assertion lengths to the v1 format.
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&MAGIC);
        buf.push(FORMAT_VERSION);
        write_u32(
            buf,
            u32::try_from(self.rows.len()).expect("statement-table row count fits v1 u32"),
        );
        for row in &self.rows {
            write_u32(buf, row.id.as_u32());
            buf.push(row.target_kind);
            buf.extend_from_slice(row.target_ref.as_bytes());
            buf.extend_from_slice(&row.target_row.to_le_bytes());
            buf.push(row.meta_kind);
            write_u32(
                buf,
                u32::try_from(row.assertion.len()).expect("statement assertion length fits v1 u32"),
            );
            buf.extend_from_slice(&row.assertion);
            buf.extend_from_slice(&row.timestamp_ns.to_le_bytes());
            buf.extend_from_slice(row.source_id.as_bytes());
            buf.extend_from_slice(&row.validity.from().as_u64().to_le_bytes());
            buf.extend_from_slice(&row.validity.to().as_u64().to_le_bytes());
        }
    }

    /// Inverse of [`Self::write_to`].
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is truncated, malformed, non-canonical, has
    /// trailing bytes, or carries an unsupported magic/version.
    pub fn read_from(data: &[u8]) -> Result<Self, String> {
        if data.len() < 9 {
            return Err("statement table too short".into());
        }
        if data[..4] != MAGIC {
            return Err("bad statement table magic".into());
        }
        let version = data[4];
        if version != FORMAT_VERSION {
            return Err(format!("unsupported statement table version {version}"));
        }
        let mut pos = 5;
        let n = read_u32(data, &mut pos)? as usize;
        let mut table = Self::new();
        for expected_index in 0..n {
            let id = LocalId::new(read_u32(data, &mut pos)?);
            let expected_id = u32::try_from(expected_index)
                .map_err(|_| "statement row count exceeds valid LocalId space")?;
            if id.as_u32() != expected_id {
                return Err(format!(
                    "non-canonical statement row id {} at index {expected_index}; expected {expected_id}",
                    id.as_u32()
                ));
            }
            let target_kind = *data.get(pos).ok_or("truncated target_kind")?;
            pos += 1;
            let target_ref = ContentId::from_bytes(read_hash(data, &mut pos)?);
            let target_row = read_u64(data, &mut pos)?;
            let meta_kind = *data.get(pos).ok_or("truncated meta_kind")?;
            pos += 1;
            let alen = read_u32(data, &mut pos)? as usize;
            let assertion_end = pos
                .checked_add(alen)
                .ok_or("statement assertion length overflow")?;
            if assertion_end > data.len() {
                return Err("truncated assertion".into());
            }
            let assertion = data[pos..assertion_end].to_vec();
            pos = assertion_end;
            let timestamp_ns = read_i64(data, &mut pos)?;
            let source_id = ContentId::from_bytes(read_hash(data, &mut pos)?);
            let from = EpochId::new(read_u64(data, &mut pos)?);
            let to = EpochId::new(read_u64(data, &mut pos)?);
            if from == EpochId::PENDING || (to != EpochId::PENDING && to <= from) {
                return Err(format!(
                    "invalid statement validity [{}, {}) at row {expected_index}",
                    from.as_u64(),
                    to.as_u64()
                ));
            }
            table.refs.intern(target_ref);
            table.refs.intern(source_id);
            table.rows.push(StatementRow {
                id,
                target_kind,
                target_ref,
                target_row,
                meta_kind,
                assertion,
                timestamp_ns,
                source_id,
                validity: EpochInterval::closed(from, to),
            });
        }
        if pos != data.len() {
            return Err(format!(
                "statement table has {} trailing bytes",
                data.len() - pos
            ));
        }
        Ok(table)
    }
}

const MAGIC: [u8; 4] = *b"GSTT";
const FORMAT_VERSION: u8 = 1;

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_i64(data: &[u8], pos: &mut usize) -> Result<i64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated i64".into());
    }
    let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_hash(data: &[u8], pos: &mut usize) -> Result<[u8; 32], String> {
    if *pos + 32 > data.len() {
        return Err("truncated hash".into());
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&data[*pos..*pos + 32]);
    *pos += 32;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: u8) -> StatementIngest {
        let mut target = [0u8; 32];
        target[0] = n;
        let mut source = [0u8; 32];
        source[0] = 0xA0;
        source[1] = n;
        StatementIngest {
            target_kind: 1,
            target_ref: target,
            target_row: u64::from(n),
            meta_kind: 2,
            assertion: vec![n, n + 1, n + 2],
            timestamp_ns: 1_000 + i64::from(n),
            source_id: source,
        }
    }

    #[test]
    fn insert_batch_is_queryable_at_tx_and_valid_time() {
        let mut table = StatementTable::new();
        let ids = table.insert_batch(&[sample(7)], EpochId::new(10));
        assert_eq!(ids, vec![LocalId::new(0)]);
        assert_eq!(table.statements_at_epoch(EpochId::new(15)).len(), 1);
        assert!(table.statements_at_epoch(EpochId::new(5)).is_empty());
        assert_eq!(table.statements_at_valid_time(1_007).len(), 1);
        assert!(table.statements_at_valid_time(1_000).is_empty());
        let target = ContentId::from_bytes({
            let mut t = [0u8; 32];
            t[0] = 7;
            t
        });
        assert_eq!(table.statements_for_target(target).len(), 1);
    }

    #[test]
    fn same_batch_twice_serializes_byte_identical() {
        let batch = [sample(1), sample(2)];
        let mut a = StatementTable::new();
        let mut b = StatementTable::new();
        a.insert_batch(&batch, EpochId::new(3));
        b.insert_batch(&batch, EpochId::new(3));
        let mut bytes_a = Vec::new();
        let mut bytes_b = Vec::new();
        a.write_to(&mut bytes_a);
        b.write_to(&mut bytes_b);
        assert_eq!(bytes_a, bytes_b, "same ingest must be byte-identical");

        let restored = StatementTable::read_from(&bytes_a).unwrap();
        let mut bytes_c = Vec::new();
        restored.write_to(&mut bytes_c);
        assert_eq!(
            bytes_a, bytes_c,
            "serialize round-trip must be byte-identical"
        );
        assert_eq!(restored.statements_at_valid_time(1_001).len(), 1);
        assert_eq!(restored.statements_at_epoch(EpochId::new(3)).len(), 2);
    }

    #[test]
    fn hashes_are_mirrored_not_reinterpreted() {
        let mut odd = [0u8; 32];
        odd[31] = 0xEE;
        let ingest = StatementIngest {
            target_kind: 9,
            target_ref: odd,
            target_row: 42,
            meta_kind: 0,
            assertion: b"opaque".to_vec(),
            timestamp_ns: 50,
            source_id: odd,
        };
        let mut table = StatementTable::new();
        table.insert_batch(&[ingest], EpochId::INITIAL);
        let row = &table.statements_at_epoch(EpochId::PENDING)[0];
        assert_eq!(row.target_ref.as_bytes(), &odd);
        assert_eq!(row.source_id.as_bytes(), &odd);
    }

    #[test]
    fn checked_insert_refuses_pending_epoch_without_mutation() {
        let mut table = StatementTable::new();
        let error = table
            .try_insert_batch(&[sample(1)], EpochId::PENDING)
            .unwrap_err();
        assert_eq!(error, StatementTableInsertError::PendingEpoch);
        assert!(table.is_empty());
    }

    #[test]
    fn decoder_rejects_noncanonical_ids_trailing_bytes_and_invalid_validity() {
        let mut table = StatementTable::new();
        table.insert_batch(&[sample(1)], EpochId::new(3));
        let mut canonical = Vec::new();
        table.write_to(&mut canonical);

        let mut bad_id = canonical.clone();
        bad_id[9..13].copy_from_slice(&1_u32.to_le_bytes());
        assert!(
            StatementTable::read_from(&bad_id)
                .unwrap_err()
                .contains("non-canonical statement row id")
        );

        let mut trailing = canonical.clone();
        trailing.push(0);
        assert!(
            StatementTable::read_from(&trailing)
                .unwrap_err()
                .contains("trailing bytes")
        );

        let mut invalid_validity = canonical;
        let validity_start = invalid_validity.len() - 16;
        invalid_validity[validity_start..validity_start + 8]
            .copy_from_slice(&EpochId::PENDING.as_u64().to_le_bytes());
        assert!(
            StatementTable::read_from(&invalid_validity)
                .unwrap_err()
                .contains("invalid statement validity")
        );
    }
}
