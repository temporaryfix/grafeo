/// Error handling for the Grafeo Dart binding.
///
/// Status codes match the C enum in `grafeo-c/src/error.rs` exactly.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'ffi/bindings.dart';

/// Status codes returned by grafeo-c FFI functions.
///
/// Values must match the C `GrafeoStatus` enum:
///   Ok=0, ErrorDatabase=1, ErrorQuery=2, ErrorTransaction=3, ErrorStorage=4,
///   ErrorIo=5, ErrorSerialization=6, ErrorInternal=7, ErrorNullPointer=8,
///   ErrorInvalidUtf8=9.
///   ErrorCancelled=10, ErrorDeadline=11, ErrorResourceLimit=12.
enum GrafeoStatus {
  ok(0),
  database(1),
  query(2),
  transaction(3),
  storage(4),
  io(5),
  serialization(6),
  internal(7),
  nullPointer(8),
  invalidUtf8(9),
  cancelled(10),
  deadline(11),
  resourceLimit(12);

  final int code;
  const GrafeoStatus(this.code);

  static GrafeoStatus fromCode(int code) =>
      values.firstWhere((s) => s.code == code, orElse: () => internal);
}

/// Base exception for all Grafeo errors.
sealed class GrafeoException implements Exception {
  final String message;
  final GrafeoStatus status;
  final String? code;
  final GrafeoException? cleanup;
  const GrafeoException(this.message, this.status, {this.code, this.cleanup});

  @override
  String toString() => '$runtimeType(${status.name}): $message';
}

/// A query parsing or execution error (status 2).
class QueryException extends GrafeoException {
  const QueryException(super.message, super.status,
      {super.code, super.cleanup});
}

/// A transaction error such as conflict or invalid state (status 3).
class TransactionException extends GrafeoException {
  const TransactionException(super.message, super.status,
      {super.code, super.cleanup});
}

/// A storage or IO error (status 4, 5).
class StorageException extends GrafeoException {
  const StorageException(super.message, super.status,
      {super.code, super.cleanup});
}

/// A serialization error (status 6).
class SerializationException extends GrafeoException {
  const SerializationException(super.message, super.status,
      {super.code, super.cleanup});
}

/// A generic database error (status 1, 7, 8, 9, or unknown).
class DatabaseException extends GrafeoException {
  /// Creates a [DatabaseException] with [message] and [status].
  const DatabaseException(super.message, super.status,
      {super.code, super.cleanup});
}

/// Map a C status code and error message to a typed Dart exception.
///
/// Mirrors `grafeo-bindings-common::error::classify_error`.
GrafeoException classifyError(int statusCode, String message,
    {String? code, GrafeoException? cleanup}) {
  final status = switch (code) {
    'GRAFEO-Q007' => GrafeoStatus.cancelled,
    'GRAFEO-Q003' => GrafeoStatus.deadline,
    'GRAFEO-S001' => GrafeoStatus.resourceLimit,
    _ => GrafeoStatus.fromCode(statusCode),
  };
  return switch (status) {
    GrafeoStatus.query ||
    GrafeoStatus.cancelled ||
    GrafeoStatus.deadline =>
      QueryException(message, status, code: code, cleanup: cleanup),
    GrafeoStatus.transaction =>
      TransactionException(message, status, code: code, cleanup: cleanup),
    GrafeoStatus.resourceLimit ||
    GrafeoStatus.storage ||
    GrafeoStatus.io =>
      StorageException(message, status, code: code, cleanup: cleanup),
    GrafeoStatus.serialization =>
      SerializationException(message, status, code: code, cleanup: cleanup),
    _ => DatabaseException(message, status, code: code, cleanup: cleanup),
  };
}

/// Read the last error message from the grafeo-c thread-local error slot.
///
/// The returned pointer is owned by the C library (thread-local storage) and
/// is valid until the next FFI call on this thread. We copy it to a Dart
/// string immediately and do NOT free it.
String lastError(GrafeoBindings bindings) {
  final ptr = bindings.grafeoLastError();
  if (ptr == nullptr) return 'Unknown error';
  return ptr.toDartString();
}

/// Throw a [GrafeoException] for a failed FFI call that returned a status code.
Never throwStatus(GrafeoBindings bindings, int statusCode) {
  throw captureError(bindings, statusCode);
}

/// Throw a [GrafeoException] for a failed FFI call that returned null.
Never throwLastError(GrafeoBindings bindings) {
  throw captureError(bindings);
}

/// Copies both native thread-local fields before leaving the executing isolate.
GrafeoException captureError(GrafeoBindings bindings, [int statusCode = 1]) {
  final message = lastError(bindings);
  final pointer = bindings.grafeoLastErrorCode();
  return classifyError(statusCode, message,
      code: pointer == nullptr ? null : pointer.toDartString());
}

GrafeoException retainCleanup(
        GrafeoException primary, GrafeoException cleanup) =>
    classifyError(primary.status.code, primary.message,
        code: primary.code, cleanup: cleanup);
