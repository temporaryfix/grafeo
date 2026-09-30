//! Language-agnostic error classification for bindings.
//!
//! Each binding maps [`ErrorCategory`] to its language-specific exception type
//! (Python `PyErr`, Node.js `napi::Error`, C `GrafeoStatus`, etc.) using a
//! single small match expression.

use grafeo_common::utils::error::Error;

/// Categories that all bindings map errors into.
///
/// These mirror the natural groupings in [`grafeo_common::utils::error::Error`]
/// and match what every binding was already doing independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Query parsing, semantic, or execution error.
    Query,
    /// Transaction conflict, timeout, or invalid state.
    Transaction,
    /// Storage-layer error (disk, memory limit).
    Storage,
    /// I/O error (file, network).
    Io,
    /// Serialization/deserialization failure.
    Serialization,
    /// Internal error (should not happen in normal operation).
    Internal,
    /// Catch-all for other database errors (not found, type mismatch, etc.).
    Database,
}

/// Classifies a Grafeo error into a binding-agnostic category.
#[must_use]
pub fn classify_error(err: &Error) -> ErrorCategory {
    match err {
        Error::Query(_) => ErrorCategory::Query,
        Error::Transaction(_) => ErrorCategory::Transaction,
        Error::Storage(_) => ErrorCategory::Storage,
        Error::Crypto(_) => ErrorCategory::Storage,
        Error::Io(_) => ErrorCategory::Io,
        Error::Serialization(_) => ErrorCategory::Serialization,
        Error::Internal(_) => ErrorCategory::Internal,
        Error::Context { source, .. } => classify_error(source),
        Error::RetainedContext { source, code } => source
            .inspect::<grafeo_common::utils::error::RetainedErrorContext, _>(|context| {
                context.primary().map(classify_error)
            })
            .flatten()
            .unwrap_or_else(|| classify_retained_code(*code)),
        _ => ErrorCategory::Database,
    }
}

fn classify_retained_code(code: grafeo_common::utils::error::ErrorCode) -> ErrorCategory {
    use grafeo_common::utils::error::ErrorCode;
    match code {
        ErrorCode::QuerySyntax
        | ErrorCode::QuerySemantic
        | ErrorCode::QueryTimeout
        | ErrorCode::QueryUnsupported
        | ErrorCode::QueryOptimization
        | ErrorCode::QueryExecution
        | ErrorCode::QueryCancelled => ErrorCategory::Query,
        ErrorCode::TransactionConflict
        | ErrorCode::TransactionTimeout
        | ErrorCode::TransactionReadOnly
        | ErrorCode::TransactionInvalidState
        | ErrorCode::TransactionSerialization
        | ErrorCode::TransactionDeadlock => ErrorCategory::Transaction,
        ErrorCode::StorageFull
        | ErrorCode::StorageCorrupted
        | ErrorCode::StorageRecoveryFailed
        | ErrorCode::CursorInvalid
        | ErrorCode::CursorForeign
        | ErrorCode::CursorEvicted => ErrorCategory::Storage,
        ErrorCode::IoError => ErrorCategory::Io,
        ErrorCode::SerializationError => ErrorCategory::Serialization,
        ErrorCode::Internal => ErrorCategory::Internal,
        _ => ErrorCategory::Database,
    }
}

/// Returns the human-readable message for a Grafeo error.
#[must_use]
pub fn error_message(err: &Error) -> String {
    err.to_string()
}

#[cfg(test)]
mod tests {
    use grafeo_common::utils::error::{
        Error, QueryError, QueryErrorKind, StorageError, TransactionError,
    };

    use super::*;

    #[test]
    fn classifies_query_error() {
        let err = Error::Query(QueryError::new(QueryErrorKind::Syntax, "bad syntax"));
        assert_eq!(classify_error(&err), ErrorCategory::Query);
    }

    #[test]
    fn timeout_and_cancellation_retain_query_category_through_context() {
        for query_error in [QueryError::timeout(), QueryError::cancelled()] {
            let direct = Error::Query(query_error.clone());
            assert_eq!(classify_error(&direct), ErrorCategory::Query);

            let contextual = Error::Query(query_error).with_context("cleanup also failed");
            assert_eq!(classify_error(&contextual), ErrorCategory::Query);
        }
    }

    #[test]
    fn retained_cleanup_keeps_primary_binding_category_and_code() {
        use grafeo_common::memory::buffer::{
            AccountedErrorPublisher, BufferManager, BufferManagerConfig, MemoryRegion,
        };
        use grafeo_common::utils::error::RetainedErrorContext;

        let manager = BufferManager::new(BufferManagerConfig::default());
        for (primary, category) in [
            (Error::Query(QueryError::cancelled()), ErrorCategory::Query),
            (Error::Query(QueryError::timeout()), ErrorCategory::Query),
            (Error::Storage(StorageError::Full), ErrorCategory::Storage),
            (
                Error::Transaction(TransactionError::Conflict),
                ErrorCategory::Transaction,
            ),
        ] {
            let code = primary.error_code();
            let publisher = AccountedErrorPublisher::try_new(
                manager
                    .try_allocate(0, MemoryRegion::ExecutionBuffers)
                    .unwrap(),
            )
            .unwrap();
            let error = Error::RetainedContext {
                code,
                source: publisher
                    .publish(RetainedErrorContext::new(
                        primary,
                        Error::Io(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
                        "cleanup",
                    ))
                    .into(),
            };
            assert!(manager.allocated() > 0);
            assert_eq!(classify_error(&error), category);
            assert_eq!(error.error_code(), code);
            drop(error);
            assert_eq!(manager.allocated(), 0);
        }
        // Core operator failures retain their original typed carrier rather
        // than being copied into the common secondary-context payload.
        let publisher = AccountedErrorPublisher::try_new(
            manager
                .try_allocate(0, MemoryRegion::ExecutionBuffers)
                .unwrap(),
        )
        .unwrap();
        let error = Error::RetainedContext {
            code: grafeo_common::utils::error::ErrorCode::QueryCancelled,
            source: publisher.publish(QueryError::cancelled()).into(),
        };
        assert_eq!(classify_error(&error), ErrorCategory::Query);
        drop(error);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn classifies_not_found_as_database() {
        let err = Error::NodeNotFound(grafeo_common::types::NodeId(42));
        assert_eq!(classify_error(&err), ErrorCategory::Database);
    }

    #[test]
    fn classifies_internal() {
        let err = Error::Internal("oops".into());
        assert_eq!(classify_error(&err), ErrorCategory::Internal);
    }

    #[test]
    fn classifies_transaction_error() {
        let err = Error::Transaction(TransactionError::Conflict);
        assert_eq!(classify_error(&err), ErrorCategory::Transaction);
    }

    #[test]
    fn classifies_storage_error() {
        let err = Error::Storage(StorageError::Full);
        assert_eq!(classify_error(&err), ErrorCategory::Storage);
    }

    #[test]
    fn classifies_io_error() {
        let err = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert_eq!(classify_error(&err), ErrorCategory::Io);
    }

    #[test]
    fn classifies_serialization_error() {
        let err = Error::Serialization("bad bytes".into());
        assert_eq!(classify_error(&err), ErrorCategory::Serialization);
    }

    #[test]
    fn error_message_is_non_empty() {
        let err = Error::Internal("something broke".into());
        let msg = error_message(&err);
        assert!(!msg.is_empty());
        assert!(msg.contains("something broke"));
    }
}
