/// Bounded invocation ownership and genuinely asynchronous native execution.
library;

import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'ffi/bindings.dart';
import 'ffi/loader.dart';
import 'types.dart';
import 'value.dart';

/// Explicit zero limits are real zero limits. Null eager limits use one million
/// rows and 64 MiB. Streams have no total row cap unless [maxRows] is supplied.
/// [maxBytes] is the combined native/managed copy envelope: one quarter is
/// admitted natively before commit, and the remainder covers managed copies.
class ExecutionOptions {
  final QueryControl? control;
  final int? maxRows;
  final int? maxBytes;
  final String? language;
  const ExecutionOptions(
      {this.control, this.maxRows, this.maxBytes, this.language});
}

class _Resources {
  final GrafeoBindings bindings;
  Pointer<Void> owner;
  Pointer<Void> cancellation;
  Pointer<GrafeoQueryOptions> options = nullptr;
  Pointer<Utf8> language = nullptr;
  _Resources(this.bindings, this.owner, this.cancellation);
  void close() {
    if (owner != nullptr) bindings.grafeoQueryControlFree(owner);
    if (cancellation != nullptr) bindings.grafeoCancelHandleFree(cancellation);
    if (language != nullptr) malloc.free(language);
    if (options != nullptr) calloc.free(options);
    owner = nullptr;
    cancellation = nullptr;
    language = nullptr;
    options = nullptr;
  }
}

/// Single-use execution owner. The timeout starts at construction, not execute.
/// Use [cancel] from the caller isolate while an asynchronous query is running.
/// Closing this wrapper never frees an invocation's independently retained owner.
class QueryControl implements Finalizable {
  final _Resources _resources;
  bool _consumed = false;
  bool _closed = false;
  static final _finalizer =
      Finalizer<_Resources>((resources) => resources.close());

  factory QueryControl({Duration? timeout, String? libraryPath}) {
    if (timeout != null && timeout.isNegative) {
      throw ArgumentError.value(timeout, 'timeout', 'must be nonnegative');
    }
    final bindings = GrafeoBindings(loadNativeLibrary(libraryPath));
    final micros = timeout?.inMicroseconds;
    final milliseconds =
        micros == null ? -1 : micros ~/ 1000 + (micros % 1000 == 0 ? 0 : 1);
    final owner = bindings.grafeoQueryControlCreate(milliseconds);
    if (owner == nullptr) throw captureError(bindings);
    final cancellation = bindings.grafeoQueryControlCancelHandle(owner);
    if (cancellation == nullptr) {
      final error = captureError(bindings);
      bindings.grafeoQueryControlFree(owner);
      throw error;
    }
    return QueryControl._(_Resources(bindings, owner, cancellation));
  }

  QueryControl._(this._resources) {
    _finalizer.attach(this, _resources, detach: this);
  }

  void cancel() {
    if (_closed) throw StateError('Query control is closed');
    final status = _resources.bindings.grafeoCancel(_resources.cancellation);
    if (status != 0) throw captureError(_resources.bindings, status);
  }

  _Resources _begin(GrafeoBindings bindings) {
    if (_closed) throw StateError('Query control is closed');
    if (_consumed) throw StateError('Query control has already been consumed');
    if (_resources.bindings.library.handle.address !=
        bindings.library.handle.address) {
      throw ArgumentError(
          'Control and execution must use the same native library');
    }
    final clone = bindings.grafeoCancelHandleClone(_resources.cancellation);
    if (clone == nullptr) throw captureError(bindings);
    final resources = _Resources(bindings, _resources.owner, clone);
    _resources.owner = nullptr;
    _consumed = true;
    return resources;
  }

  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer.detach(this);
    _resources.close();
  }
}

/// Internal bridge retained until every worker using its addresses has exited.
class NativeInvocation implements Finalizable {
  final _Resources _resources;
  final int copyBytes;
  final int collectRows;
  int _active = 0;
  bool _closed = false;
  static final _finalizer =
      Finalizer<_Resources>((resources) => resources.close());

  Pointer<GrafeoQueryOptions> get options => _resources.options;
  GrafeoBindings get bindings => _resources.bindings;

  NativeInvocation._(this._resources, this.copyBytes, this.collectRows) {
    _finalizer.attach(this, _resources, detach: this);
  }

  factory NativeInvocation.create(GrafeoBindings bindings,
      {ExecutionOptions? options, bool streaming = false}) {
    final maxRows = options?.maxRows;
    final bytes = options?.maxBytes ?? 64 * 1024 * 1024;
    final language = options?.language;
    final nativeMax = sizeOf<IntPtr>() == 8 ? 0x7fffffffffffffff : 0xffffffff;
    if (bytes < 0 ||
        bytes > nativeMax ||
        (maxRows != null && (maxRows < 0 || maxRows > nativeMax))) {
      throw ArgumentError('Limits must be nonnegative and fit native size_t');
    }
    if (language != null) validateExecutionText(language, 'language');
    // Enforce the contiguous managed capacity in native precommit admission.
    final nativeBytes = (bytes ~/ 4).clamp(0, 0x7fffffff).toInt();
    QueryControl? temporary;
    _Resources? resources;
    try {
      final control = options?.control;
      if (control == null) {
        final owner = bindings.grafeoQueryControlCreate(-1);
        if (owner == nullptr) throw captureError(bindings);
        final cancel = bindings.grafeoQueryControlCancelHandle(owner);
        if (cancel == nullptr) {
          final error = captureError(bindings);
          bindings.grafeoQueryControlFree(owner);
          throw error;
        }
        temporary = QueryControl._(_Resources(bindings, owner, cancel));
      }
      resources = (control ?? temporary!)._begin(bindings);
      resources.options = calloc<GrafeoQueryOptions>();
      if (language != null && language.isNotEmpty) {
        resources.language = language.toNativeUtf8(allocator: malloc);
      }
      resources.options.ref
        ..control = resources.owner
        ..maxRows = maxRows ??
            (streaming ? (sizeOf<IntPtr>() == 8 ? -1 : 0xffffffff) : 1000000)
        ..maxBytes = nativeBytes
        ..language = resources.language;
      return NativeInvocation._(
          resources, bytes - nativeBytes, maxRows ?? 1000000);
    } catch (_) {
      resources?.close();
      rethrow;
    } finally {
      temporary?.close();
    }
  }

  void beginWork() {
    if (_closed) throw StateError('Invocation is closed');
    _active++;
  }

  void endWork() {
    if (_active == 0) throw StateError('No active invocation worker');
    _active--;
  }

  void cancel() {
    if (!_closed) bindings.grafeoCancel(_resources.cancellation);
  }

  void close() {
    if (_closed) return;
    if (_active != 0) {
      cancel();
      throw StateError('Invocation worker must exit before close');
    }
    _closed = true;
    _finalizer.detach(this);
    _resources.close();
  }
}

/// Reject C-string truncation and malformed UTF-16 before consuming a control.
void validateNativeQueryText(String text) => validateExecutionText(text);

/// Validates native text with an optional field name for diagnostics.
void validateExecutionText(String text, [String name = 'query']) {
  for (var i = 0; i < text.length; i++) {
    final unit = text.codeUnitAt(i);
    if (unit == 0) throw ArgumentError('$name contains NUL');
    if (unit >= 0xd800 && unit <= 0xdbff) {
      if (++i >= text.length ||
          text.codeUnitAt(i) < 0xdc00 ||
          text.codeUnitAt(i) > 0xdfff) {
        throw ArgumentError('$name contains invalid UTF-16');
      }
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      throw ArgumentError('$name contains invalid UTF-16');
    }
  }
}

StorageException copyLimitError(String message) =>
    StorageException(message, GrafeoStatus.resourceLimit, code: 'GRAFEO-S001');

/// Borrows native JSON and admits all conversion copies before allocating.
class ManagedCopyBudget {
  int remaining;
  ManagedCopyBudget(this.remaining);
  void charge(int amount) {
    if (amount < 0 || amount > remaining) {
      throw copyLimitError('Managed copies exceed byte envelope');
    }
    remaining -= amount;
  }

  String readUtf8(Pointer<Utf8> pointer) {
    if (pointer == nullptr) {
      throw const DatabaseException(
          'Null native JSON', GrafeoStatus.nullPointer);
    }
    final data = pointer.cast<Uint8>();
    var length = 0;
    var cost = 0;
    var inString = false;
    var escaped = false;
    while (data[length] != 0) {
      if (length == 0x7fffffff) {
        throw copyLimitError('Native JSON exceeds managed capacity');
      }
      final byte = data[length++];
      cost += 6;
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (byte == 92) {
          escaped = true;
        } else if (byte == 34) {
          inString = false;
        }
      } else {
        switch (byte) {
          case 34:
            inString = true;
            cost += 64;
          case 123:
            cost += 576;
          case 58:
            cost += 320;
          case 91:
            cost += 128;
          case 44:
            cost += 96;
        }
      }
      if (cost > remaining) {
        throw copyLimitError('Managed JSON exceeds byte envelope');
      }
    }
    charge(cost);
    return pointer.toDartString(length: length);
  }
}

/// Owns and frees the native result even when conversion fails.
QueryResult decodeQueryResult(
    GrafeoBindings bindings, Pointer<Void> result, int copyBytes) {
  try {
    final budget = ManagedCopyBudget(copyBytes);
    final json = budget.readUtf8(bindings.grafeoResultJson(result));
    final time = bindings.grafeoResultExecutionTimeMs(result);
    final scanned = bindings.grafeoResultRowsScanned(result);
    final rows = parseRows(json);
    final (nodes, edges) = extractEntities(rows);
    return QueryResult(
        columns: extractColumns(rows),
        rows: rows,
        nodes: nodes,
        edges: edges,
        executionTimeMs: time,
        rowsScanned: scanned);
  } finally {
    bindings.grafeoFreeResult(result);
  }
}

/// Executes on a worker isolate and joins its exit before releasing any owner.
/// The caller must retain its database/transaction until this future completes.
Future<QueryResult> runNativeExecution(
    {required String? libraryPath,
    required int handleAddress,
    required String query,
    String? paramsJson,
    required NativeInvocation invocation,
    bool transaction = false}) async {
  invocation.beginWork();
  final port = ReceivePort();
  final exited = Completer<void>();
  Object? response;
  final subscription = port.listen((dynamic message) {
    if (message == null) {
      if (!exited.isCompleted) exited.complete();
    } else {
      response = message;
    }
  });
  try {
    await Isolate.spawn<Map<String, Object?>>(
        _nativeExecutionWorker,
        {
          'port': port.sendPort,
          'libraryPath': libraryPath,
          'handle': handleAddress,
          'query': query,
          'params': paramsJson,
          'options': invocation.options.address,
          'copyBytes': invocation.copyBytes,
          'transaction': transaction,
        },
        onExit: port.sendPort,
        onError: port.sendPort,
        errorsAreFatal: true);
    await exited.future;
    final message = response;
    if (message is QueryResult) return message;
    if (message is Map) {
      throw classifyError(
          message['status'] as int, message['message'] as String,
          code: message['code'] as String?);
    }
    throw DatabaseException(
        'Native worker failed: $message', GrafeoStatus.internal);
  } finally {
    await subscription.cancel();
    port.close();
    invocation.endWork();
  }
}

// This entry point receives only sendable descriptors, never Finalizable owners.
void _nativeExecutionWorker(Map<String, Object?> request) {
  final reply = request['port'] as SendPort;
  Object response;
  try {
    final bindings =
        GrafeoBindings(loadNativeLibrary(request['libraryPath'] as String?));
    final query = (request['query'] as String).toNativeUtf8(allocator: malloc);
    final text = request['params'] as String?;
    final params = text?.toNativeUtf8(allocator: malloc) ?? nullptr;
    try {
      final handle = Pointer<Void>.fromAddress(request['handle'] as int);
      final options =
          Pointer<GrafeoQueryOptions>.fromAddress(request['options'] as int);
      final result = request['transaction'] == true
          ? bindings.grafeoTransactionExecuteWithOptions(
              handle, query, params, options)
          : bindings.grafeoExecuteWithOptions(handle, query, params, options);
      if (result == nullptr) throw captureError(bindings);
      response =
          decodeQueryResult(bindings, result, request['copyBytes'] as int);
    } finally {
      malloc.free(query);
      if (params != nullptr) malloc.free(params);
    }
  } catch (error) {
    response = error is GrafeoException
        ? {
            'status': error.status.code,
            'message': error.message,
            'code': error.code
          }
        : {
            'status': GrafeoStatus.internal.code,
            'message': error.toString(),
            'code': null
          };
  }
  // All native cleanup is complete. Transfer managed output without another copy.
  Isolate.exit(reply, response);
}
