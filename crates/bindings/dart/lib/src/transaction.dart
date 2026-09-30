/// ACID transaction support with native ownership retained across worker isolates.
library;

import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'error.dart';
import 'execution.dart';
import 'ffi/bindings.dart';
import 'types.dart';
import 'value.dart';

/// An ACID transaction. Unfinished native work is rolled back on finalization.
class Transaction implements Finalizable {
  final GrafeoBindings _bindings;
  final String? _libraryPath;
  // Bound database methods retain the parent even between transaction calls.
  final void Function()? _reserveParent;
  final void Function()? _releaseParent;
  Pointer<Void> _handle;
  bool _finished = false;
  bool _busy = false;
  static final Map<int, NativeFinalizer> _finalizers = {};
  late final NativeFinalizer _finalizer;

  /// Wrap a native transaction. Usually created through GrafeoDB.beginTransaction.
  Transaction(
    this._handle,
    this._bindings, {
    String? libraryPath,
    void Function()? reserveParent,
    void Function()? releaseParent,
  })  : _libraryPath = libraryPath,
        _reserveParent = reserveParent,
        _releaseParent = releaseParent {
    _finalizer = _finalizers.putIfAbsent(
        _bindings.library.handle.address,
        () => NativeFinalizer(
              _bindings.library
                  .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
                      'grafeo_free_transaction'),
            ));
    _finalizer.attach(this, _handle.cast(), detach: this);
  }

  void _reserve() {
    if (_finished) {
      throw TransactionException(
          'Transaction already finished', GrafeoStatus.transaction);
    }
    if (_busy) {
      throw TransactionException(
          'Transaction is busy', GrafeoStatus.transaction);
    }
    _busy = true;
    try {
      _reserveParent?.call();
    } catch (_) {
      _busy = false;
      rethrow;
    }
  }

  void _release() {
    try {
      _releaseParent?.call();
    } finally {
      _busy = false;
    }
  }

  /// Execute with explicit query ownership, cancellation authority, and limits.
  QueryResult executeWithOptions(
    String query, {
    ExecutionOptions? options,
    Map<String, dynamic>? params,
  }) {
    _reserve();
    NativeInvocation? invocation;
    Pointer<Utf8> queryPtr = nullptr;
    Pointer<Utf8> paramsPtr = nullptr;
    try {
      validateNativeQueryText(query);
      final paramsJson = params == null ? null : encodeParams(params);
      if (paramsJson != null) validateNativeQueryText(paramsJson);
      invocation = NativeInvocation.create(_bindings, options: options);
      queryPtr = query.toNativeUtf8(allocator: malloc);
      if (paramsJson != null) {
        paramsPtr = paramsJson.toNativeUtf8(allocator: malloc);
      }
      final result = _bindings.grafeoTransactionExecuteWithOptions(
          _handle, queryPtr, paramsPtr, invocation.options);
      if (result == nullptr) throwLastError(_bindings);
      return decodeQueryResult(_bindings, result, invocation.copyBytes);
    } finally {
      malloc.free(queryPtr);
      malloc.free(paramsPtr);
      try {
        invocation?.close();
      } finally {
        _release();
      }
    }
  }

  /// Execute on a worker isolate. Cancel through [ExecutionOptions.control].
  /// Reserve this transaction and its parent before serialization or scheduling.
  Future<QueryResult> executeWithOptionsAsync(
    String query, {
    ExecutionOptions? options,
    Map<String, dynamic>? params,
  }) {
    _reserve();
    NativeInvocation? invocation;
    try {
      validateNativeQueryText(query);
      final paramsJson = params == null ? null : encodeParams(params);
      if (paramsJson != null) validateNativeQueryText(paramsJson);
      final owner = NativeInvocation.create(_bindings, options: options);
      invocation = owner;
      return runNativeExecution(
              libraryPath: _libraryPath,
              handleAddress: _handle.address,
              query: query,
              paramsJson: paramsJson,
              invocation: owner,
              transaction: true)
          .whenComplete(() {
        try {
          owner.close();
        } finally {
          _release();
        }
      });
    } catch (_) {
      try {
        invocation?.close();
      } finally {
        _release();
      }
      rethrow;
    }
  }

  /// Execute a GQL query.
  QueryResult execute(String query) => executeWithOptions(query);

  /// Execute a GQL query on a worker isolate.
  Future<QueryResult> executeAsync(String query, {ExecutionOptions? options}) =>
      executeWithOptionsAsync(query, options: options);

  /// Execute a GQL query with typed parameters.
  QueryResult executeWithParams(String query, Map<String, dynamic> params) =>
      executeWithOptions(query, params: params);

  /// Execute a parameterized query on a worker isolate.
  Future<QueryResult> executeWithParamsAsync(
          String query, Map<String, dynamic> params,
          {ExecutionOptions? options}) =>
      executeWithOptionsAsync(query, params: params, options: options);

  /// Execute a query in the given language with optional typed parameters.
  QueryResult executeLanguage(String language, String query,
          {Map<String, dynamic>? params}) =>
      executeWithOptions(query,
          options: ExecutionOptions(language: language), params: params);

  /// Commit and release the native transaction on success.
  void commit() => _complete(true);

  /// Roll back and release the native transaction on success.
  void rollback() => _complete(false);

  void _complete(bool commit) {
    _reserve();
    try {
      final status = commit
          ? _bindings.grafeoCommit(_handle)
          : _bindings.grafeoRollback(_handle);
      // Capture native error state before cleanup; a failed commit remains owned.
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      _finished = true;
      _finalizer.detach(this);
      _bindings.grafeoFreeTransaction(_handle);
      _handle = nullptr;
    } finally {
      _release();
    }
  }
}
