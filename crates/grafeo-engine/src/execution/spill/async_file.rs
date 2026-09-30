//! Async adapters for the shared framed operation owners.

use super::async_manager::{AsyncSpillError, Scheduler};
use grafeo_core::execution::spill::{
    OwnedSpillFile, OwnedSpillReader, OwnedSpillRecord, OwnedSpillWrite, SpillFileIdentity,
};
use std::sync::Arc;

/// A move-only framed file. A dropped operation waiter permanently consumes
/// this handle; the physical worker retains it until the operation settles.
pub struct AsyncSpillFile {
    file: Option<OwnedSpillFile>,
    scheduler: Arc<Scheduler>,
    identity: SpillFileIdentity,
}

#[allow(
    clippy::result_large_err,
    reason = "operation failures retain their accounting owners"
)]
impl AsyncSpillFile {
    pub(super) fn from_owned(file: OwnedSpillFile, scheduler: Arc<Scheduler>) -> Self {
        Self {
            identity: file.identity(),
            file: Some(file),
            scheduler,
        }
    }

    /// Immutable file identity supplied to the record provider.
    #[must_use]
    pub const fn identity(&self) -> SpillFileIdentity {
        self.identity
    }

    /// Schedules one admitted, bounded record operation, including publication.
    ///
    /// # Errors
    /// Returns the original owned I/O failure before cancellation/cleanup errors.
    pub async fn write(&mut self, operation: OwnedSpillWrite) -> Result<(), AsyncSpillError> {
        let file = self.file.take().ok_or(AsyncSpillError::Closed)?;
        self.file = Some(
            self.scheduler
                .run(false, move || file.write(operation).map_err(Into::into))
                .await?,
        );
        Ok(())
    }

    /// Opens the shared qualified reader after publication.
    ///
    /// # Errors
    /// Returns publication, admission, cancellation or provider errors.
    pub async fn reader(&mut self) -> Result<AsyncSpillFileReader, AsyncSpillError> {
        let file = self.file.take().ok_or(AsyncSpillError::Closed)?;
        let (reader, file) = self
            .scheduler
            .run(false, move || {
                let reader = file.reader()?;
                Ok((reader, file))
            })
            .await?;
        self.file = Some(file);
        Ok(AsyncSpillFileReader {
            reader: Some(reader),
            scheduler: Arc::clone(&self.scheduler),
        })
    }

    /// Closes and deletes through the retained file capability, even after cancellation.
    ///
    /// # Errors
    /// Returns a live-reader, cleanup or scheduling error without releasing quota early.
    pub async fn close_and_delete(&mut self) -> Result<(), AsyncSpillError> {
        let file = self.file.take().ok_or(AsyncSpillError::Closed)?;
        self.scheduler
            .run(true, move || file.close_and_delete().map_err(Into::into))
            .await
    }
}

/// Reader jobs own their provider workspace and exact decoded output grants.
pub struct AsyncSpillFileReader {
    reader: Option<OwnedSpillReader>,
    scheduler: Arc<Scheduler>,
}

#[allow(
    clippy::result_large_err,
    reason = "operation failures retain their accounting owners"
)]
impl AsyncSpillFileReader {
    /// Reads a sort or partition declaration; partition columns are zero.
    ///
    /// # Errors
    /// Returns framing, resource or cancellation failures.
    pub async fn read_declaration(&mut self) -> Result<(u32, u64), AsyncSpillError> {
        let reader = self.reader.take().ok_or(AsyncSpillError::Closed)?;
        let (reader, declaration) = self
            .scheduler
            .run(false, move || reader.read_declaration().map_err(Into::into))
            .await?;
        self.reader = Some(reader);
        Ok(declaration)
    }

    /// Reads one exact record whose grant remains attached to the returned bytes.
    ///
    /// # Errors
    /// Returns framing, resource, provider or cancellation failures.
    pub async fn read_record(&mut self) -> Result<OwnedSpillRecord, AsyncSpillError> {
        let reader = self.reader.take().ok_or(AsyncSpillError::Closed)?;
        let (reader, record) = self
            .scheduler
            .run(false, move || reader.read_record().map_err(Into::into))
            .await?;
        self.reader = Some(reader);
        Ok(record)
    }

    /// Validates terminal framing and closes the reader.
    ///
    /// # Errors
    /// Returns terminal framing, I/O or accounting-release errors.
    pub async fn finish(&mut self) -> Result<(), AsyncSpillError> {
        let reader = self.reader.take().ok_or(AsyncSpillError::Closed)?;
        self.scheduler
            .run(false, move || reader.finish().map_err(Into::into))
            .await
    }

    /// Closes without validating terminal framing; remains available after cancellation.
    ///
    /// # Errors
    /// Returns physical cleanup or accounting-release errors.
    pub async fn close(&mut self) -> Result<(), AsyncSpillError> {
        let reader = self.reader.take().ok_or(AsyncSpillError::Closed)?;
        self.scheduler
            .run(true, move || reader.close().map_err(Into::into))
            .await
    }
}
