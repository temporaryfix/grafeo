/// Bounded native read cursors with explicit, fallible cleanup.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'execution.dart';
import 'ffi/bindings.dart';
import 'ffi/loader.dart';
import 'types.dart';
import 'value.dart';

/// Pulls one bounded row or chunk. Use [closeAsync] to interrupt and join an
/// active asynchronous pull. Terminal errors remain observable after close.
class ResultStream implements Finalizable {
  final GrafeoBindings _bindings;
  final NativeInvocation _invocation;
  final String? _libraryPath;
  // Retain the wrapper whose raw database pointer opened this cursor.
  final Object? _parentOwner;
  final List<String> _columns;
  late final NativeFinalizer _finalizer;
  static final _finalizers = <int, NativeFinalizer>{};
  Pointer<Void> _handle;
  bool _closed = false;
  bool _terminal = false;
  bool _closing = false;
  Future<void>? _pending;
  Object? _failure;
  StackTrace? _failureStack;
  Object? _cleanupError;

  ResultStream._(this._handle, this._columns, this._bindings, this._invocation,
      this._libraryPath, this._parentOwner) {
    _finalizer = _finalizers.putIfAbsent(
        _bindings.library.handle.address,
        () => NativeFinalizer(_bindings.library
            .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
                'grafeo_stream_free')));
    _finalizer.attach(this, _handle, detach: this);
  }

  /// Opens a cursor through the same single-use control as eager queries.
  static ResultStream open(
      Pointer<Void> dbHandle, GrafeoBindings bindings, String query,
      {ExecutionOptions? options,
      Map<String, dynamic>? params,
      String? libraryPath,
      Object? parentOwner}) {
    validateNativeQueryText(query);
    final paramsJson = params == null ? null : encodeParams(params);
    if (paramsJson != null) validateNativeQueryText(paramsJson);
    final invocation =
        NativeInvocation.create(bindings, options: options, streaming: true);
    Pointer<Utf8> queryPtr = nullptr;
    Pointer<Utf8> paramsPtr = nullptr;
    Pointer<Void> handle = nullptr;
    try {
      queryPtr = query.toNativeUtf8(allocator: malloc);
      if (paramsJson != null) {
        paramsPtr = paramsJson.toNativeUtf8(allocator: malloc);
      }
      handle = bindings.grafeoStreamOpenWithOptions(
          dbHandle, queryPtr, paramsPtr, invocation.options);
      if (handle == nullptr) throwLastError(bindings);
      final columnsPtr = bindings.grafeoStreamColumnsJson(handle);
      if (columnsPtr == nullptr) throwLastError(bindings);
      final List<String> columns;
      try {
        final json =
            ManagedCopyBudget(invocation.copyBytes).readUtf8(columnsPtr);
        columns = List<String>.unmodifiable(
            (jsonDecode(json) as List).cast<String>());
      } finally {
        bindings.grafeoFreeString(columnsPtr);
      }
      return ResultStream._(
          handle, columns, bindings, invocation, libraryPath, parentOwner);
    } catch (error, stack) {
      Object failure = error;
      if (handle != nullptr) {
        final status = bindings.grafeoStreamClose(handle);
        if (status != 0) {
          final cleanup = captureError(bindings, status);
          failure = error is GrafeoException
              ? retainCleanup(error, cleanup)
              : StreamCleanupException(error, cleanup);
        }
        bindings.grafeoStreamFree(handle);
      }
      invocation.close();
      Error.throwWithStackTrace(failure, stack);
    } finally {
      malloc.free(queryPtr);
      malloc.free(paramsPtr);
    }
  }

  List<String> get columns => _columns;

  /// A secondary native close failure, if a pull or decoder already failed.
  Object? get cleanupError => _cleanupError;

  void _check() {
    _throwFailure();
    if (_closed || _closing) {
      throw const DatabaseException(
          'ResultStream is closed', GrafeoStatus.database);
    }
    if (_pending != null) {
      throw const DatabaseException(
          'ResultStream has an active pull', GrafeoStatus.database);
    }
  }

  void _throwFailure() {
    if (_failure case final Object error) {
      Error.throwWithStackTrace(error, _failureStack ?? StackTrace.current);
    }
  }

  void _finish([Object? primary, StackTrace? stack]) {
    if (!_terminal) {
      _terminal = true;
      final status = _bindings.grafeoStreamClose(_handle);
      Object? cleanup;
      if (status != 0) {
        try {
          throwStatus(_bindings, status);
        } catch (error) {
          cleanup = error;
        }
      }
      _finalizer.detach(this);
      _bindings.grafeoStreamFree(_handle);
      _handle = nullptr;
      _invocation.close();
      _failure = primary ?? cleanup;
      _failureStack = stack ?? StackTrace.current;
      if (primary != null) _cleanupError = cleanup;
    } else if (_failure == null && primary != null) {
      _failure = primary;
      _failureStack = stack;
    }
    _throwFailure();
  }

  Map<String, dynamic>? next() =>
      _next(ManagedCopyBudget(_invocation.copyBytes));

  Map<String, dynamic>? _next(ManagedCopyBudget budget) {
    _check();
    if (_terminal) return null;
    try {
      final row = _readRow(_bindings, _handle, budget);
      if (row == null) _finish();
      return row;
    } catch (error, stack) {
      _finish(error, stack);
      rethrow;
    }
  }

  /// Returns at most [maxRows] (native maximum 1024).
  QueryResult? nextChunk(int maxRows) {
    if (maxRows <= 0) throw RangeError.value(maxRows, 'maxRows');
    _check();
    if (_terminal) return null;
    try {
      final result =
          _readChunk(_bindings, _handle, maxRows, _invocation.copyBytes);
      if (result == null) _finish();
      return result;
    } catch (error, stack) {
      _finish(error, stack);
      rethrow;
    }
  }

  Future<Object?> _pullAsync(int maxRows) async {
    _check();
    if (_terminal) return null;
    final completion = Completer<void>();
    _pending = completion.future;
    _invocation.beginWork();
    Object? result;
    Object? failure;
    StackTrace? failureStack;
    try {
      // Only primitive job data crosses isolates. The native owners remain
      // reachable here until the worker has exited and transferred its result.
      result = await _runPull(
          (_libraryPath, _handle.address, maxRows, _invocation.copyBytes));
    } catch (error, stack) {
      failure = error;
      failureStack = stack;
    } finally {
      _invocation.endWork();
      _pending = null;
      completion.complete();
    }
    if (failure != null) _finish(failure, failureStack);
    if (result == null) _finish();
    return result;
  }

  Future<Map<String, dynamic>?> nextAsync() async =>
      await _pullAsync(0) as Map<String, dynamic>?;

  Future<QueryResult?> nextChunkAsync(int maxRows) async {
    if (maxRows <= 0) throw RangeError.value(maxRows, 'maxRows');
    return await _pullAsync(maxRows) as QueryResult?;
  }

  /// Dart synchronous iterators have no disposal callback. Close explicitly
  /// after an early break; exhaustion and native failures release the cursor.
  Iterable<Map<String, dynamic>> rows() sync* {
    while (true) {
      final row = next();
      if (row == null) return;
      yield row;
    }
  }

  /// Pulls bounded chunks off the event loop, and closes on subscription cancel.
  Stream<Map<String, dynamic>> rowsAsync() async* {
    try {
      while (true) {
        final chunk = await nextChunkAsync(1024);
        if (chunk == null) return;
        for (final row in chunk.rows) {
          yield row;
        }
      }
    } finally {
      await closeAsync();
    }
  }

  /// Collects within total row and byte caps. Failure returns no partial list.
  List<Map<String, dynamic>> toList() {
    final budget = ManagedCopyBudget(_invocation.copyBytes);
    var buffer = <Map<String, dynamic>>[];
    var count = 0;
    while (true) {
      final row = _next(budget);
      if (row == null) break;
      if (count >= _invocation.collectRows) {
        _collectionLimit('Collection exceeds row cap');
      }
      if (count == buffer.length) {
        if (count >= 0x0fffffff) {
          _collectionLimit('Collection exceeds managed array capacity');
        }
        final capacity =
            count == 0 ? 4 : (count * 2).clamp(0, 0x0fffffff).toInt();
        _chargeCollection(budget, capacity * sizeOf<IntPtr>() + 64);
        final grown = List<Map<String, dynamic>>.filled(capacity, row);
        grown.setRange(0, count, buffer);
        buffer = grown;
      }
      buffer[count++] = row;
    }
    if (count == buffer.length) return buffer;
    _chargeCollection(budget, count * sizeOf<IntPtr>() + 64);
    return List<Map<String, dynamic>>.generate(count, (index) => buffer[index],
        growable: false);
  }

  void _chargeCollection(ManagedCopyBudget budget, int amount) {
    if (amount > budget.remaining) {
      _collectionLimit('Collection exceeds byte cap');
    }
    budget.remaining -= amount;
  }

  Never _collectionLimit(String message) {
    final error = copyLimitError(message);
    _finish(error, StackTrace.current);
    throw error;
  }

  /// Closes idle cursors. During an async pull, requests cancellation and reports
  /// busy without freeing the live pointer; await [closeAsync] to join it.
  void close() {
    if (_pending != null) {
      _closing = true;
      _invocation.cancel();
      throw const DatabaseException(
          'ResultStream is busy; await closeAsync', GrafeoStatus.database);
    }
    if (_closed) {
      _throwFailure();
      return;
    }
    _closing = _closed = true;
    _finish();
    // Keep the parent owner reachable through the last native operation.
    assert(_parentOwner == null || _parentOwner is Object);
  }

  Future<void> closeAsync() async {
    _closing = true;
    final pending = _pending;
    if (pending != null) {
      _invocation.cancel();
      await pending;
    }
    close();
  }
}

Map<String, dynamic>? _readRow(
    GrafeoBindings bindings, Pointer<Void> handle, ManagedCopyBudget budget) {
  final out = calloc<Pointer<Utf8>>();
  try {
    final status = bindings.grafeoStreamNextRowJson(handle, out);
    if (status != 0) throwStatus(bindings, status);
    final pointer = out.value;
    if (pointer == nullptr) return null;
    try {
      final decoded = jsonDecode(budget.readUtf8(pointer));
      if (decoded is! Map<String, dynamic>) {
        throw const SerializationException(
            'Native row is not an object', GrafeoStatus.serialization);
      }
      return decoded;
    } finally {
      bindings.grafeoFreeString(pointer);
    }
  } finally {
    calloc.free(out);
  }
}

QueryResult? _readChunk(
    GrafeoBindings bindings, Pointer<Void> handle, int maxRows, int copyBytes) {
  final out = calloc<Pointer<Void>>();
  try {
    final status = bindings.grafeoStreamNextChunk(handle, maxRows, out);
    if (status != 0) throwStatus(bindings, status);
    if (out.value == nullptr) return null;
    return decodeQueryResult(bindings, out.value, copyBytes);
  } finally {
    calloc.free(out);
  }
}

typedef _PullJob = (String?, int, int, int);

// A separate top-level scope prevents closures from capturing Finalizable owners.
Future<Object?> _runPull(_PullJob job) => Isolate.run(() => _executePull(job));
Object? _executePull(_PullJob job) {
  final (libraryPath, address, maxRows, copyBytes) = job;
  final bindings = GrafeoBindings(loadNativeLibrary(libraryPath));
  final handle = Pointer<Void>.fromAddress(address);
  return maxRows == 0
      ? _readRow(bindings, handle, ManagedCopyBudget(copyBytes))
      : _readChunk(bindings, handle, maxRows, copyBytes);
}

/// Preserves a managed opener failure together with a native cleanup failure.
class StreamCleanupException implements Exception {
  final Object cause;
  final GrafeoException cleanup;
  StreamCleanupException(this.cause, this.cleanup);
  @override
  String toString() => '$cause (cleanup: $cleanup)';
}
