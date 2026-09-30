//! Claim / statement ingest on [`GrafeoDB`] (Track A3).

use grafeo_common::types::{ContentId, EpochId, LocalId};
use grafeo_common::utils::error::{Result, TransactionError};
use grafeo_core::graph::compact::statement_table::{StatementIngest, StatementRow};

use super::GrafeoDB;

impl GrafeoDB {
    /// Appends a ClaimRef-layout batch as in-memory temporal statement rows.
    ///
    /// Hashes (`target_ref`, `source_id`) are stored opaquely. `tx_epoch` is
    /// transaction time; `timestamp_ns` on each row is valid/phenomenon time.
    ///
    /// This is the `temporal-host` compile slice, not the authoritative
    /// RDF statement path. It is deliberately refused on persistent databases
    /// until the table participates in the shared Session/WAL/checkpoint
    /// protocol. Silently accepting the batch there would acknowledge a write
    /// that cannot survive reopen.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error for persistent databases, read-only
    /// databases, or the uncommitted [`EpochId::PENDING`] sentinel.
    pub fn insert_statement_batch(
        &self,
        batch: &[StatementIngest],
        tx_epoch: EpochId,
    ) -> Result<Vec<LocalId>> {
        if self.is_read_only() {
            return Err(TransactionError::ReadOnly.into());
        }
        if self.is_persistent() {
            return Err(TransactionError::InvalidState(
                "insert_statement_batch is an in-memory temporal-host preview; persistent statement writes must use the RDF Session/WAL path"
                    .to_string(),
            )
            .into());
        }
        if tx_epoch == EpochId::PENDING {
            return Err(TransactionError::InvalidState(
                "insert_statement_batch requires a committed transaction epoch".to_string(),
            )
            .into());
        }
        self.statement_table
            .write()
            .try_insert_batch(batch, tx_epoch)
            .map_err(|error| TransactionError::InvalidState(error.to_string()).into())
    }

    /// Statements whose transaction-time interval contains `epoch`.
    #[must_use]
    pub fn statements_at_epoch(&self, epoch: EpochId) -> Vec<StatementRow> {
        self.statement_table
            .read()
            .statements_at_epoch(epoch)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Statements whose valid-time `timestamp_ns` equals `timestamp_ns`.
    #[must_use]
    pub fn statements_at_valid_time(&self, timestamp_ns: i64) -> Vec<StatementRow> {
        self.statement_table
            .read()
            .statements_at_valid_time(timestamp_ns)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Statements whose opaque `target_ref` matches `target`.
    #[must_use]
    pub fn statements_for_target(&self, target: ContentId) -> Vec<StatementRow> {
        self.statement_table
            .read()
            .statements_for_target(target)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Deterministic statement-table bytes (A3.2).
    #[must_use]
    pub fn serialize_statement_table(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.statement_table.read().write_to(&mut buf);
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::graph::compact::statement_table::{StatementIngest, StatementTable};

    fn row(n: u8) -> StatementIngest {
        let mut target = [0u8; 32];
        target[0] = n;
        StatementIngest {
            target_kind: 1,
            target_ref: target,
            target_row: u64::from(n),
            meta_kind: 0,
            assertion: vec![n],
            timestamp_ns: 100 + i64::from(n),
            source_id: [n; 32],
        }
    }

    #[test]
    fn engine_insert_is_queryable_and_deterministic() {
        let db_a = GrafeoDB::new_in_memory();
        let db_b = GrafeoDB::new_in_memory();
        let batch = [row(1), row(2)];
        let ids = db_a
            .insert_statement_batch(&batch, EpochId::new(10))
            .unwrap();
        db_b.insert_statement_batch(&batch, EpochId::new(10))
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(db_a.statements_at_epoch(EpochId::new(15)).len(), 2);
        assert_eq!(db_a.statements_at_valid_time(101).len(), 1);
        assert_eq!(
            db_a.serialize_statement_table(),
            db_b.serialize_statement_table()
        );
        let restored = StatementTable::read_from(&db_a.serialize_statement_table()).unwrap();
        let mut again = Vec::new();
        restored.write_to(&mut again);
        assert_eq!(again, db_a.serialize_statement_table());
    }

    #[test]
    fn pending_epoch_is_refused_without_mutation() {
        let db = GrafeoDB::new_in_memory();
        let error = db
            .insert_statement_batch(&[row(1)], EpochId::PENDING)
            .unwrap_err();
        assert!(matches!(
            error,
            grafeo_common::utils::error::Error::Transaction(TransactionError::InvalidState(_))
        ));
        assert!(db.statements_at_epoch(EpochId::PENDING).is_empty());
    }

    #[cfg(feature = "wal")]
    #[test]
    fn persistent_preview_write_is_refused_without_mutation() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = GrafeoDB::with_config(crate::Config::persistent(dir.path().join("statements")))
            .unwrap();
        let error = db
            .insert_statement_batch(&[row(1)], EpochId::new(10))
            .unwrap_err();
        assert!(matches!(
            error,
            grafeo_common::utils::error::Error::Transaction(TransactionError::InvalidState(_))
        ));
        assert!(db.statements_at_epoch(EpochId::PENDING).is_empty());
    }
}
