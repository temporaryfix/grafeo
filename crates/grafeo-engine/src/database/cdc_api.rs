//! Model-independent database-level Change Data Capture API.
//!
//! CDC is shared by LPG, RDF, and mixed-model sessions. Keeping these methods
//! outside the LPG-only admin module makes that contract true at the feature
//! boundary as well as at runtime.

use grafeo_common::utils::error::Result;

impl super::GrafeoDB {
    /// Reads a bounded page from the durable retained native feed.
    ///
    /// `None` starts at the current retained floor. A returned cursor resumes
    /// exclusively after its last event and survives exact checkpoint/reopen or
    /// snapshot transfer preserving StoreId. Forks have a different StoreId.
    /// An in-memory database retains positions only for its process lifetime;
    /// reopening durability requires an authoritative persisted cut.
    /// Both limits must be positive. `max_bytes` bounds the sum of bincode standard
    /// encodings of the returned events (excluding the page/cursor envelope).
    /// Accounting streams without allocating an intermediate serialized image.
    /// The row bound also caps owned event slots; no whole-feed copy is built.
    ///
    /// # Errors
    /// Returns structured invalid, foreign, or evicted cursor errors, resource
    /// exhaustion when the next event alone exceeds the byte limit, invalid input
    /// for zero limits, or a durability error when recovery is required.
    pub fn changes_after(
        &self,
        cursor: Option<&grafeo_common::types::DurableCursor>,
        max_events: usize,
        max_bytes: usize,
    ) -> Result<crate::cdc::ChangePage> {
        if !*self.is_open.read() {
            return Err(grafeo_common::utils::error::TransactionError::InvalidState(
                "database is closed".into(),
            )
            .into());
        }
        let _publication = self.transaction_manager.publication().read();
        self.check_cdc_readable()?;
        self.cdc_log.page(
            (
                self.store_id(),
                grafeo_common::types::FeedId::new(self.config.graph_model.as_u8() + 1, 0)?,
                self.transaction_manager.current_epoch(),
            ),
            None,
            cursor,
            max_events,
            max_bytes,
            |_| true,
        )
    }

    fn check_cdc_readable(&self) -> Result<()> {
        if self.is_durability_poisoned() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::DurabilityFailure(
                    "CDC history is unavailable after a durability failure; reopen and recover before reading it"
                        .to_string(),
                ),
            ));
        }
        Ok(())
    }

    /// Returns whether CDC is enabled by default for new sessions.
    #[must_use]
    pub fn is_cdc_enabled(&self) -> bool {
        self.cdc_active()
    }

    /// Sets whether CDC is enabled by default for new sessions.
    ///
    /// Does not affect sessions that were already created.
    pub fn set_cdc_enabled(&self, enabled: bool) {
        self.cdc_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Installs a database-scoped pause immediately before transactional CDC
    /// insertion. Test builds use this to observe the publication barrier at
    /// its final state/event cut; dropping the returned authority resets it.
    ///
    /// # Errors
    ///
    /// Returns an error if this exact database already has an installed pause.
    #[cfg(feature = "testing-statement-injection")]
    #[doc(hidden)]
    pub fn testing_pause_cdc_before_publication(
        &self,
    ) -> std::result::Result<crate::cdc::CdcPublicationPause, &'static str> {
        self.cdc_log.install_publication_pause()
    }

    /// Fails this database's next CDC publication preparation before its marker.
    #[cfg(feature = "testing-statement-injection")]
    #[doc(hidden)]
    pub fn testing_fail_next_cdc_preparation(&self) {
        self.cdc_log.fail_next_preparation();
    }

    /// Returns whether the mixed-model publication barrier is currently held
    /// for writing by this exact database.
    ///
    /// This deterministic test probe qualifies the CDC pause cut without
    /// relying on reader-thread scheduling or timeout timing.
    #[cfg(feature = "testing-statement-injection")]
    #[doc(hidden)]
    #[must_use]
    pub fn testing_publication_write_locked(&self) -> bool {
        self.transaction_manager.publication().try_read().is_none()
    }

    /// Reads bounded indexed history using the shared durable feed cursor.
    ///
    /// The row limit caps inspected entity candidates, including hidden or
    /// filtered rows. Epoch/graph predicates run before payload accounting.
    /// An exhausted entity index may advance an empty page to the shared tail;
    /// an unchanged cursor marks EOF. Widening the query requires an earlier
    /// cursor or `None`. Both limits are positive and required.
    ///
    /// # Errors
    /// Returns lifecycle, durability, cursor, permission or resource errors.
    #[inline]
    pub fn history_after(
        &self,
        query: &crate::cdc::EntityHistoryQuery,
        cursor: Option<&grafeo_common::types::DurableCursor>,
        max_events: usize,
        max_bytes: usize,
    ) -> Result<crate::cdc::ChangePage> {
        if !*self.is_open.read() {
            return Err(grafeo_common::utils::error::TransactionError::InvalidState(
                "database is closed".into(),
            )
            .into());
        }
        let _publication = self.transaction_manager.publication().read();
        self.check_cdc_readable()?;
        self.cdc_log.page(
            (
                self.store_id(),
                grafeo_common::types::FeedId::new(self.config.graph_model.as_u8() + 1, 0)?,
                self.transaction_manager.current_epoch(),
            ),
            Some(query.entity_id),
            cursor,
            max_events,
            max_bytes,
            |event| query.matches(event),
        )
    }

    #[cfg(all(test, feature = "cdc", feature = "lpg"))]
    fn history_fixture(
        &self,
        query: crate::cdc::EntityHistoryQuery,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        let mut cursor = None;
        let mut events = Vec::new();
        loop {
            let page = self.history_after(&query, cursor.as_ref(), 2, 64 * 1024 * 1024)?;
            if cursor == Some(page.next) {
                break;
            }
            cursor = Some(page.next);
            events.extend(page.events);
        }
        Ok(events)
    }

    #[cfg(all(test, feature = "cdc", feature = "lpg"))]
    pub(crate) fn history(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        self.history_fixture(crate::cdc::EntityHistoryQuery {
            entity_id: entity_id.into(),
            graph: crate::cdc::HistoryGraph::All,
            since_epoch: grafeo_common::types::EpochId::INITIAL,
        })
    }

    #[cfg(all(test, feature = "cdc", feature = "lpg"))]
    pub(crate) fn history_in_graph(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
        graph: &grafeo_common::types::GraphPath,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        self.history_fixture(crate::cdc::EntityHistoryQuery {
            entity_id: entity_id.into(),
            graph: crate::cdc::HistoryGraph::Lpg(graph.clone()),
            since_epoch: grafeo_common::types::EpochId::INITIAL,
        })
    }

    // Collection is available only to finite unit-test fixtures. Production
    // callers consume required row/byte bounds through changes_after.
    #[cfg(test)]
    pub(crate) fn changes_between(
        &self,
        start_epoch: grafeo_common::types::EpochId,
        end_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        let mut cursor = None;
        let mut events = Vec::new();
        loop {
            let page = self.changes_after(cursor.as_ref(), 2, 64 * 1024 * 1024)?;
            if cursor == Some(page.next) {
                break;
            }
            cursor = Some(page.next);
            events.extend(
                page.events
                    .into_iter()
                    .filter(|event| event.epoch >= start_epoch && event.epoch <= end_epoch),
            );
        }
        Ok(events)
    }
}
