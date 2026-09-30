/// The main GrafeoDB database class.
///
/// Wraps the grafeo-c shared library via FFI. Uses [NativeFinalizer] to
/// prevent leaks if [close] is not called explicitly.
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'cdc.dart';
import 'error.dart';
import 'execution.dart';
import 'ffi/bindings.dart';
import 'ffi/loader.dart';
import 'index_request.dart';
import 'stream.dart';
import 'transaction.dart';
import 'types.dart';
import 'value.dart';

/// A Grafeo graph database instance.
///
/// Create with [GrafeoDB.memory] (in-memory) or [GrafeoDB.open] (persistent).
/// Always call [close] when done, or rely on [NativeFinalizer] as a safety net.
class GrafeoDB implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  bool _closed = false;
  int _activeCalls = 0;
  final String? _libraryPath;

  static final Map<int, NativeFinalizer> _finalizers = {};
  late final NativeFinalizer _finalizer;

  GrafeoDB._(this._handle, this._bindings, this._libraryPath) {
    // Lazily create a finalizer that calls grafeo_free_database on the handle.
    // The Rust Drop impl for the inner Arc<RwLock<GrafeoDB>> flushes writes.
    _finalizer = _finalizers.putIfAbsent(
        _bindings.library.handle.address,
        () => NativeFinalizer(
              _bindings.library
                  .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
                'grafeo_free_database',
              ),
            ));
    _finalizer.attach(this, _handle.cast(), detach: this);
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /// Create a new in-memory database.
  static GrafeoDB memory({String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final ptr = bindings.grafeoOpenMemory();
    if (ptr == nullptr) throwLastError(bindings);
    return GrafeoDB._(ptr, bindings, libraryPath);
  }

  /// Open a persistent database at [path].
  static GrafeoDB open(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpen(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings, libraryPath);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Open or create a single-file `.grafeo` database at [path].
  ///
  /// Recommended for embedded use (desktop apps, mobile apps). All data is
  /// stored in one file with a sidecar WAL for crash safety, similar to
  /// DuckDB's `.duckdb` format.
  static GrafeoDB openSingleFile(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpenSingleFile(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings, libraryPath);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Open an existing database at [path] in read-only mode.
  ///
  /// Multiple read-only handles may be opened concurrently on the same path.
  /// Write operations on a read-only database will throw [DatabaseException].
  static GrafeoDB openReadOnly(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpenReadOnly(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings, libraryPath);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Close the database, flushing all writes.
  ///
  /// Safe to call multiple times. After close, all other methods throw.
  void close() {
    if (_closed) return;
    if (_activeCalls != 0) {
      throw DatabaseException('Database is busy', GrafeoStatus.database);
    }
    final status = _bindings.grafeoClose(_handle);
    if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    _closed = true;
    _finalizer.detach(this);
    _bindings.grafeoFreeDatabase(_handle);
    _handle = nullptr;
  }

  /// Returns the grafeo-c library version.
  ///
  /// The C function returns a pointer to a static string that must NOT
  /// be freed (it lives in the binary's read-only data segment).
  static String version({String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final ptr = bindings.grafeoVersion();
    return ptr.toDartString();
  }

  void _checkOpen() {
    if (_closed) {
      throw DatabaseException('Database is closed', GrafeoStatus.database);
    }
  }

  // ===========================================================================
  // Query execution
  // ===========================================================================

  /// Execute a query with explicit ownership and result limits.
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
      final result = _bindings.grafeoExecuteWithOptions(
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
  /// The handle and invocation are reserved before serialization or scheduling.
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
              invocation: owner)
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

  /// Execute a query in the requested language.
  QueryResult executeLanguage(String language, String query,
          {Map<String, dynamic>? params}) =>
      executeWithOptions(query,
          options: ExecutionOptions(language: language), params: params);

  /// Open a bounded lazy cursor over a read-only query.
  ResultStream executeStream(String query) => executeStreamWithOptions(query);

  /// Open a bounded cursor with query ownership and optional parameters.
  ResultStream executeStreamWithOptions(
    String query, {
    ExecutionOptions? options,
    Map<String, dynamic>? params,
  }) {
    _reserve();
    try {
      return ResultStream.open(_handle, _bindings, query,
          options: options,
          params: params,
          libraryPath: _libraryPath,
          parentOwner: this);
    } finally {
      _release();
    }
  }

  void _reserve() {
    _checkOpen();
    _activeCalls++;
  }

  void _release() {
    _activeCalls--;
  }

  /// Execute a Cypher query.
  QueryResult executeCypher(String query) => executeLanguage('cypher', query);

  /// Execute a Cypher query with typed parameters.
  QueryResult executeCypherWithParams(
          String query, Map<String, dynamic> params) =>
      executeLanguage('cypher', query, params: params);

  /// Execute a Gremlin query.
  QueryResult executeGremlin(String query) => executeLanguage('gremlin', query);

  /// Execute a Gremlin query with typed parameters.
  QueryResult executeGremlinWithParams(
          String query, Map<String, dynamic> params) =>
      executeLanguage('gremlin', query, params: params);

  /// Execute a Graphql query.
  QueryResult executeGraphql(String query) => executeLanguage('graphql', query);

  /// Execute a Graphql query with typed parameters.
  QueryResult executeGraphqlWithParams(
          String query, Map<String, dynamic> params) =>
      executeLanguage('graphql', query, params: params);

  /// Execute a Sparql query.
  QueryResult executeSparql(String query) => executeLanguage('sparql', query);

  /// Execute a Sparql query with typed parameters.
  QueryResult executeSparqlWithParams(
          String query, Map<String, dynamic> params) =>
      executeLanguage('sparql', query, params: params);

  // ===========================================================================
  // Statistics and info
  // ===========================================================================

  /// Get the number of nodes in the database (O(1), synchronous).
  int get nodeCount {
    _reserve();
    try {
      return _bindings.grafeoNodeCount(_handle);
    } finally {
      _release();
    }
  }

  /// Get the number of edges in the database (O(1), synchronous).
  int get edgeCount {
    _reserve();
    try {
      return _bindings.grafeoEdgeCount(_handle);
    } finally {
      _release();
    }
  }

  /// Get database info as a parsed JSON map.
  ///
  /// The C function allocates a string that must be freed with
  /// grafeo_free_string.
  Map<String, dynamic> info() {
    _reserve();
    try {
      final ptr = _bindings.grafeoInfo(_handle);
      if (ptr == nullptr) throwLastError(_bindings);
      try {
        return parseObject(ptr.toDartString());
      } finally {
        _bindings.grafeoFreeString(ptr);
      }
    } finally {
      _release();
    }
  }

  // ===========================================================================
  // Schema context
  // ===========================================================================

  /// Set the active schema for subsequent execute calls on this database.
  ///
  /// Equivalent to running `SESSION SET SCHEMA 'schemaName'` via GQL, but
  /// without requiring a round-trip query. All queries issued after this call
  /// will be scoped to [schemaName] until [resetSchema] is called.
  ///
  /// Throws [DatabaseException] if the schema does not exist.
  void setSchema(String schemaName) {
    _reserve();
    try {
      final schemaPtr = schemaName.toNativeUtf8(allocator: malloc);
      try {
        final status = _bindings.grafeoSetSchema(_handle, schemaPtr);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      } finally {
        malloc.free(schemaPtr);
      }
    } finally {
      _release();
    }
  }

  /// Clear the active schema context, reverting to the default graph store.
  void resetSchema() {
    _reserve();
    try {
      final status = _bindings.grafeoResetSchema(_handle);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      _release();
    }
  }

  /// Return the currently active schema name, or `null` if none is set.
  String? currentSchema() {
    _reserve();
    try {
      final ptr = _bindings.grafeoCurrentSchema(_handle);
      if (ptr == nullptr) return null;
      try {
        return ptr.toDartString();
      } finally {
        _bindings.grafeoFreeString(ptr);
      }
    } finally {
      _release();
    }
  }

  // ===========================================================================
  // Transactions
  // ===========================================================================

  /// Begin a new transaction with the default isolation level.
  Transaction beginTransaction() {
    _reserve();
    try {
      final txPtr = _bindings.grafeoBeginTransaction(_handle);
      if (txPtr == nullptr) throwLastError(_bindings);
      return Transaction(txPtr, _bindings,
          libraryPath: _libraryPath,
          reserveParent: _reserve,
          releaseParent: _release);
    } finally {
      _release();
    }
  }

  /// Begin a transaction with a specific [isolationLevel].
  Transaction beginTransactionWithIsolation(IsolationLevel isolationLevel) {
    _reserve();
    try {
      final txPtr = _bindings.grafeoBeginTransactionWithIsolation(
        _handle,
        isolationLevel.code,
      );
      if (txPtr == nullptr) throwLastError(_bindings);
      return Transaction(txPtr, _bindings,
          libraryPath: _libraryPath,
          reserveParent: _reserve,
          releaseParent: _release);
    } finally {
      _release();
    }
  }

  /// Controls capture for subsequent sessions.
  bool get cdcEnabled {
    _reserve();
    try {
      return _bindings.grafeoIsCdcEnabled(_handle);
    } finally {
      _release();
    }
  }

  set cdcEnabled(bool enabled) {
    _reserve();
    try {
      _bindings.grafeoSetCdcEnabled(_handle, enabled);
    } finally {
      _release();
    }
  }

  /// Reads an owned page. Null starts at the retained floor; other cursors have
  /// exactly 97 bytes. Limits are positive; bytes count native event encodings,
  /// excluding JSON/page envelopes. An unchanged next cursor means EOF.
  ChangePage changesAfter(
    Uint8List? cursor, {
    required int maxEvents,
    required int maxBytes,
  }) => _readChangePage(
    cursor,
    maxEvents,
    maxBytes,
    _bindings.grafeoChangesAfter,
  );

  /// Reads node history, inclusive of [sinceEpoch]. Coordinates use exact u64s.
  ChangePage nodeHistoryAfter(
    BigInt id,
    Uint8List? cursor, {
    BigInt? sinceEpoch,
    required int maxEvents,
    required int maxBytes,
  }) => _readChangePage(
    cursor,
    maxEvents,
    maxBytes,
    (db, ptr, len, rows, bytes) => _bindings.grafeoNodeHistoryAfter(
      db,
      _u64Bits(id),
      _u64Bits(sinceEpoch ?? BigInt.zero),
      ptr,
      len,
      rows,
      bytes,
    ),
  );

  /// Reads edge history with the same bounds and ownership as [changesAfter].
  ChangePage edgeHistoryAfter(
    BigInt id,
    Uint8List? cursor, {
    BigInt? sinceEpoch,
    required int maxEvents,
    required int maxBytes,
  }) => _readChangePage(
    cursor,
    maxEvents,
    maxBytes,
    (db, ptr, len, rows, bytes) => _bindings.grafeoEdgeHistoryAfter(
      db,
      _u64Bits(id),
      _u64Bits(sinceEpoch ?? BigInt.zero),
      ptr,
      len,
      rows,
      bytes,
    ),
  );

  static int _u64Bits(BigInt value) {
    if (value.isNegative || value.bitLength > 64) {
      throw ArgumentError.value(
        value,
        'coordinate',
        'must fit unsigned 64 bits',
      );
    }
    // Dart native int is signed; Uint64 FFI passes the same 64 bits to C.
    return value.toSigned(64).toInt();
  }

  ChangePage _readChangePage(
    Uint8List? cursor,
    int maxEvents,
    int maxBytes,
    Pointer<Void> Function(Pointer<Void>, Pointer<Uint8>, int, int, int) read,
  ) {
    _reserve();
    Pointer<Uint8> input = nullptr;
    try {
      if (cursor != null) {
        // Preserve non-null empty input. Native rejects noncanonical lengths
        // before dereferencing; avoid allocating from an untrusted length.
        input = malloc<Uint8>(97);
        if (cursor.length == 97) input.asTypedList(97).setAll(0, cursor);
      }
      final page = read(
        _handle,
        input,
        cursor?.length ?? 0,
        maxEvents < 0 ? 0 : maxEvents,
        maxBytes < 0 ? 0 : maxBytes,
      );
      if (page == nullptr) throwLastError(_bindings);
      try {
        final next = Uint8List.fromList(
          _bindings.grafeoChangePageCursor(page).asTypedList(97),
        );
        final json = _bindings.grafeoChangePageEventsJson(page).toDartString();
        return ChangePage.fromJson(json, next);
      } finally {
        _bindings.grafeoFreeChangePage(page);
      }
    } finally {
      if (input != nullptr) malloc.free(input);
      _release();
    }
  }

  // ===========================================================================
  // Node CRUD
  // ===========================================================================

  /// Create a node with [labels] and [properties]. Returns the new node ID.
  int createNode(List<String> labels, Map<String, dynamic> properties) {
    _reserve();
    try {
      final labelsJson = jsonEncode(labels);
      final propsJson = encodeParams(properties);
      final labelsPtr = labelsJson.toNativeUtf8(allocator: malloc);
      final propsPtr = propsJson.toNativeUtf8(allocator: malloc);
      try {
        final id = _bindings.grafeoCreateNode(_handle, labelsPtr, propsPtr);
        if (id == -1) throwLastError(_bindings); // C returns u64::MAX on error
        return id;
      } finally {
        malloc.free(labelsPtr);
        malloc.free(propsPtr);
      }
    } finally {
      _release();
    }
  }

  /// Get a node by [id]. Returns a [Node] or throws on error.
  Node getNode(int id) {
    _reserve();
    try {
      final outPtr = malloc<Pointer<Void>>();
      try {
        final status = _bindings.grafeoGetNode(_handle, id, outPtr);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
        final nodePtr = outPtr.value;
        try {
          final nodeId = _bindings.grafeoNodeId(nodePtr);
          final labelsJson =
              _bindings.grafeoNodeLabelsJson(nodePtr).toDartString();
          final propsJson =
              _bindings.grafeoNodePropertiesJson(nodePtr).toDartString();
          return Node(
            nodeId,
            parseStringArray(labelsJson),
            parseObject(propsJson),
          );
        } finally {
          _bindings.grafeoFreeNode(nodePtr);
        }
      } finally {
        malloc.free(outPtr);
      }
    } finally {
      _release();
    }
  }

  /// Return the labels of node [id] without fetching the full node.
  ///
  /// More efficient than [getNode] when only labels are needed.
  List<String> getNodeLabels(int id) {
    _reserve();
    try {
      final ptr = _bindings.grafeoGetNodeLabels(_handle, id);
      if (ptr == nullptr) throwLastError(_bindings);
      try {
        return parseStringArray(ptr.toDartString());
      } finally {
        _bindings.grafeoFreeString(ptr);
      }
    } finally {
      _release();
    }
  }

  /// Delete a node by [id]. Returns true on success.
  bool deleteNode(int id) {
    _reserve();
    try {
      final result = _bindings.grafeoDeleteNode(_handle, id);
      if (result < 0) throwLastError(_bindings);
      return result == 1;
    } finally {
      _release();
    }
  }

  /// Set a property on node [id].
  void setNodeProperty(int id, String key, dynamic value) {
    _reserve();
    try {
      final keyPtr = key.toNativeUtf8(allocator: malloc);
      final valueJson = encodeValue(value);
      final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
      try {
        final status = _bindings.grafeoSetNodeProperty(
          _handle,
          id,
          keyPtr,
          valuePtr,
        );
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      } finally {
        malloc.free(keyPtr);
        malloc.free(valuePtr);
      }
    } finally {
      _release();
    }
  }

  /// Remove a property from node [id].
  void removeNodeProperty(int id, String key) {
    _reserve();
    try {
      final keyPtr = key.toNativeUtf8(allocator: malloc);
      try {
        final result = _bindings.grafeoRemoveNodeProperty(_handle, id, keyPtr);
        if (result < 0) throwLastError(_bindings);
      } finally {
        malloc.free(keyPtr);
      }
    } finally {
      _release();
    }
  }

  /// Add a label to node [id].
  void addNodeLabel(int id, String label) {
    _reserve();
    try {
      final labelPtr = label.toNativeUtf8(allocator: malloc);
      try {
        final result = _bindings.grafeoAddNodeLabel(_handle, id, labelPtr);
        if (result < 0) throwLastError(_bindings);
      } finally {
        malloc.free(labelPtr);
      }
    } finally {
      _release();
    }
  }

  /// Remove a label from node [id].
  void removeNodeLabel(int id, String label) {
    _reserve();
    try {
      final labelPtr = label.toNativeUtf8(allocator: malloc);
      try {
        final result = _bindings.grafeoRemoveNodeLabel(_handle, id, labelPtr);
        if (result < 0) throwLastError(_bindings);
      } finally {
        malloc.free(labelPtr);
      }
    } finally {
      _release();
    }
  }

  // ===========================================================================
  // Edge CRUD
  // ===========================================================================

  /// Create an edge from [sourceId] to [targetId] with the given [type] and
  /// [properties]. Returns the new edge ID.
  int createEdge(
    int sourceId,
    int targetId,
    String type,
    Map<String, dynamic> properties,
  ) {
    _reserve();
    try {
      final typePtr = type.toNativeUtf8(allocator: malloc);
      final propsJson = encodeParams(properties);
      final propsPtr = propsJson.toNativeUtf8(allocator: malloc);
      try {
        final id = _bindings.grafeoCreateEdge(
          _handle,
          sourceId,
          targetId,
          typePtr,
          propsPtr,
        );
        if (id == -1) throwLastError(_bindings); // C returns u64::MAX on error
        return id;
      } finally {
        malloc.free(typePtr);
        malloc.free(propsPtr);
      }
    } finally {
      _release();
    }
  }

  /// Get an edge by [id]. Returns an [Edge] or throws on error.
  Edge getEdge(int id) {
    _reserve();
    try {
      final outPtr = malloc<Pointer<Void>>();
      try {
        final status = _bindings.grafeoGetEdge(_handle, id, outPtr);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
        final edgePtr = outPtr.value;
        try {
          final edgeId = _bindings.grafeoEdgeId(edgePtr);
          final sourceId = _bindings.grafeoEdgeSourceId(edgePtr);
          final targetId = _bindings.grafeoEdgeTargetId(edgePtr);
          final edgeType = _bindings.grafeoEdgeType(edgePtr).toDartString();
          final propsJson =
              _bindings.grafeoEdgePropertiesJson(edgePtr).toDartString();
          return Edge(
            edgeId,
            edgeType,
            sourceId,
            targetId,
            parseObject(propsJson),
          );
        } finally {
          _bindings.grafeoFreeEdge(edgePtr);
        }
      } finally {
        malloc.free(outPtr);
      }
    } finally {
      _release();
    }
  }

  /// Delete an edge by [id]. Returns true on success.
  bool deleteEdge(int id) {
    _reserve();
    try {
      final result = _bindings.grafeoDeleteEdge(_handle, id);
      if (result < 0) throwLastError(_bindings);
      return result == 1;
    } finally {
      _release();
    }
  }

  /// Set a property on edge [id].
  void setEdgeProperty(int id, String key, dynamic value) {
    _reserve();
    try {
      final keyPtr = key.toNativeUtf8(allocator: malloc);
      final valueJson = encodeValue(value);
      final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
      try {
        final status = _bindings.grafeoSetEdgeProperty(
          _handle,
          id,
          keyPtr,
          valuePtr,
        );
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      } finally {
        malloc.free(keyPtr);
        malloc.free(valuePtr);
      }
    } finally {
      _release();
    }
  }

  /// Remove a property from edge [id].
  void removeEdgeProperty(int id, String key) {
    _reserve();
    try {
      final keyPtr = key.toNativeUtf8(allocator: malloc);
      try {
        final result = _bindings.grafeoRemoveEdgeProperty(_handle, id, keyPtr);
        if (result < 0) throwLastError(_bindings);
      } finally {
        malloc.free(keyPtr);
      }
    } finally {
      _release();
    }
  }

  // ===========================================================================
  // Property indexes
  // ===========================================================================

  /// Commit one catalog-owned index and return its uint32 owner.
  int createIndex(CreateIndexRequest request) {
    _reserve();
    try {
      return using((Arena arena) {
        final native = arena<GrafeoIndexRequest>();
        final value = native.ref;
        value.kind = request.kind.index;
        void writeSpan(GrafeoUtf8 target, String text) {
          for (var i = 0; i < text.length; i++) {
            final unit = text.codeUnitAt(i);
            if (unit >= 0xd800 && unit <= 0xdbff) {
              if (i + 1 >= text.length ||
                  text.codeUnitAt(i + 1) < 0xdc00 ||
                  text.codeUnitAt(i + 1) > 0xdfff) {
                throw ArgumentError('Index request contains malformed UTF-16');
              }
              i++; // Consume the valid low surrogate.
            } else if (unit >= 0xdc00 && unit <= 0xdfff) {
              throw ArgumentError('Index request contains malformed UTF-16');
            }
          }
          final bytes = utf8.encode(text);
          target.len = bytes.length;
          if (bytes.isEmpty) {
            target.data = nullptr;
          } else {
            target.data = arena<Uint8>(bytes.length);
            target.data.asTypedList(bytes.length).setAll(0, bytes);
          }
        }

        writeSpan(value.property, request.property);
        value.graphCount = request.graph.length;
        if (request.graph.isNotEmpty) {
          value.graph = arena<GrafeoUtf8>(request.graph.length);
          for (var i = 0; i < request.graph.length; i++) {
            writeSpan((value.graph + i).ref, request.graph[i]);
          }
        }
        void optionalString(int bit, GrafeoUtf8 target, String? text) {
          if (text != null) {
            value.options |= bit;
            writeSpan(target, text);
          }
        }

        int sizeValue(int number) {
          final maximum =
              sizeOf<IntPtr>() == 4 ? 0xffffffff : 0x7fffffffffffffff;
          if (number < 0 || number > maximum) {
            throw RangeError.range(number, 0, maximum, 'index option');
          }
          return number;
        }

        optionalString(1, value.name, request.name);
        optionalString(2, value.label, request.label);
        optionalString(8, value.metric, request.metric);
        optionalString(64, value.quantization, request.quantization);
        final dimensions = request.dimensions;
        if (dimensions != null) {
          value.options |= 4;
          value.dimensions = sizeValue(dimensions);
        }
        final m = request.m;
        if (m != null) {
          value.options |= 16;
          value.m = sizeValue(m);
        }
        final ef = request.efConstruction;
        if (ef != null) {
          value.options |= 32;
          value.efConstruction = sizeValue(ef);
        }
        final minimum = request.minTokenLength;
        if (minimum != null) {
          value.options |= 128;
          value.minTokenLength = sizeValue(minimum);
        }
        final owner = arena<Uint32>();
        final status = _bindings.grafeoCreateIndex(_handle, native, owner);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
        return owner.value;
      }, calloc);
    } finally {
      _release();
    }
  }

  /// Drop exactly this owner; returns false only when the owner is absent.
  bool dropIndex(int owner) {
    _reserve();
    try {
      RangeError.checkValueInInterval(owner, 0, 0xffffffff, 'owner');
      return using((Arena arena) {
        final dropped = arena<Int32>();
        final status = _bindings.grafeoDropIndex(_handle, owner, dropped);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
        return dropped.value != 0;
      }, calloc);
    } finally {
      _release();
    }
  }

  /// Rebuild this owner with its resolved configuration. Missing owners fail.
  void rebuildIndex(int owner) {
    _reserve();
    try {
      RangeError.checkValueInInterval(owner, 0, 0xffffffff, 'owner');
      final status = _bindings.grafeoRebuildIndex(_handle, owner);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      _release();
    }
  }

  /// Returns true if a property index exists for [propertyKey].
  bool hasPropertyIndex(String propertyKey) {
    _reserve();
    try {
      final keyPtr = propertyKey.toNativeUtf8(allocator: malloc);
      try {
        return _bindings.grafeoHasPropertyIndex(_handle, keyPtr) != 0;
      } finally {
        malloc.free(keyPtr);
      }
    } finally {
      _release();
    }
  }

  /// Find all node IDs where [propertyKey] equals [value].
  ///
  /// Requires that a property index exists for [propertyKey] (see
  /// [createIndex]). Returns the matching node IDs; call [getNode]
  /// to retrieve full node data.
  List<int> findNodesByProperty(String propertyKey, dynamic value) {
    _reserve();
    try {
      final keyPtr = propertyKey.toNativeUtf8(allocator: malloc);
      final valueJson = encodeValue(value);
      final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
      final outIdsPtr = malloc<Pointer<Uint64>>();
      final outCountPtr = malloc<IntPtr>();
      try {
        final status = _bindings.grafeoFindNodesByProperty(
          _handle,
          keyPtr,
          valuePtr,
          outIdsPtr,
          outCountPtr,
        );
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
        final count = outCountPtr.value;
        if (count == 0) return [];
        final ids = outIdsPtr.value;
        final result = [for (var i = 0; i < count; i++) ids[i]];
        _bindings.grafeoFreeNodeIds(ids, count);
        return result;
      } finally {
        malloc.free(keyPtr);
        malloc.free(valuePtr);
        malloc.free(outIdsPtr);
        malloc.free(outCountPtr);
      }
    } finally {
      _release();
    }
  }

  // ===========================================================================
  // Vector operations
  // ===========================================================================

  /// Perform a k-nearest-neighbour vector search.
  ///
  /// Returns up to [k] results ordered by similarity. [ef] controls the
  /// search quality vs. speed trade-off (typical: 64). For diversity-aware
  /// search, use [mmrSearch] instead.
  List<VectorResult> vectorSearch(
    String label,
    String property,
    List<double> query, {
    required int k,
    int ef = 64,
  }) {
    _reserve();
    try {
      final labelPtr = label.toNativeUtf8(allocator: malloc);
      final propertyPtr = property.toNativeUtf8(allocator: malloc);
      final queryPtr = malloc<Float>(query.length);
      for (var i = 0; i < query.length; i++) {
        queryPtr[i] = query[i];
      }
      final outIdsPtr = malloc<Pointer<Uint64>>();
      final outDistsPtr = malloc<Pointer<Float>>();
      final outCountPtr = malloc<IntPtr>();

      try {
        final status = _bindings.grafeoVectorSearch(
          _handle,
          labelPtr,
          propertyPtr,
          queryPtr,
          query.length,
          k,
          ef,
          outIdsPtr,
          outDistsPtr,
          outCountPtr,
        );
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);

        final count = outCountPtr.value;
        if (count == 0) return [];

        final ids = outIdsPtr.value;
        final dists = outDistsPtr.value;
        final results = <VectorResult>[
          for (var i = 0; i < count; i++) VectorResult(ids[i], dists[i]),
        ];

        _bindings.grafeoFreeVectorResults(ids, dists, count);
        return results;
      } finally {
        malloc.free(labelPtr);
        malloc.free(propertyPtr);
        malloc.free(queryPtr);
        malloc.free(outIdsPtr);
        malloc.free(outDistsPtr);
        malloc.free(outCountPtr);
      }
    } finally {
      _release();
    }
  }

  /// Bulk-create [vectors.length] nodes, each labelled [label], with the
  /// embedding stored under [embeddingProperty].
  ///
  /// [vectors] is a list of N equal-length float vectors (one per node).
  /// Returns the list of created node IDs in insertion order.
  ///
  /// Requires the `vector-index` feature in grafeo-c.
  List<int> batchCreateNodes(
    String label,
    String embeddingProperty,
    List<List<double>> vectors,
  ) {
    _reserve();
    try {
      if (vectors.isEmpty) return [];
      final dimensions = vectors.first.length;
      final count = vectors.length;
      final labelPtr = label.toNativeUtf8(allocator: malloc);
      final propertyPtr = embeddingProperty.toNativeUtf8(allocator: malloc);
      final vectorPtr = malloc<Float>(count * dimensions);
      for (var i = 0; i < count; i++) {
        for (var j = 0; j < dimensions; j++) {
          vectorPtr[i * dimensions + j] = vectors[i][j];
        }
      }
      final outIdsPtr = malloc<Pointer<Uint64>>();
      final outCountPtr = malloc<IntPtr>();
      try {
        final status = _bindings.grafeoBatchCreateNodes(
          _handle,
          labelPtr,
          propertyPtr,
          vectorPtr,
          count,
          dimensions,
          outIdsPtr,
          outCountPtr,
        );
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
        final resultCount = outCountPtr.value;
        if (resultCount == 0) return [];
        final ids = outIdsPtr.value;
        final result = [for (var i = 0; i < resultCount; i++) ids[i]];
        _bindings.grafeoFreeNodeIds(ids, resultCount);
        return result;
      } finally {
        malloc.free(labelPtr);
        malloc.free(propertyPtr);
        malloc.free(vectorPtr);
        malloc.free(outIdsPtr);
        malloc.free(outCountPtr);
      }
    } finally {
      _release();
    }
  }

  /// Perform an MMR (Maximal Marginal Relevance) vector search.
  List<VectorResult> mmrSearch(
    String label,
    String property,
    List<double> query, {
    required int k,
    required int fetchK,
    required double lambda,
    required int ef,
  }) {
    _reserve();
    try {
      final labelPtr = label.toNativeUtf8(allocator: malloc);
      final propertyPtr = property.toNativeUtf8(allocator: malloc);
      final queryPtr = malloc<Float>(query.length);
      for (var i = 0; i < query.length; i++) {
        queryPtr[i] = query[i];
      }
      final outIdsPtr = malloc<Pointer<Uint64>>();
      final outDistsPtr = malloc<Pointer<Float>>();
      final outCountPtr = malloc<IntPtr>();

      try {
        final status = _bindings.grafeoMmrSearch(
          _handle,
          labelPtr,
          propertyPtr,
          queryPtr,
          query.length,
          k,
          fetchK,
          lambda,
          ef,
          outIdsPtr,
          outDistsPtr,
          outCountPtr,
        );
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);

        final count = outCountPtr.value;
        if (count == 0) return [];

        final ids = outIdsPtr.value;
        final dists = outDistsPtr.value;
        final results = <VectorResult>[
          for (var i = 0; i < count; i++) VectorResult(ids[i], dists[i]),
        ];

        _bindings.grafeoFreeVectorResults(ids, dists, count);
        return results;
      } finally {
        malloc.free(labelPtr);
        malloc.free(propertyPtr);
        malloc.free(queryPtr);
        malloc.free(outIdsPtr);
        malloc.free(outDistsPtr);
        malloc.free(outCountPtr);
      }
    } finally {
      _release();
    }
  }

  // ===========================================================================
  // Admin
  // ===========================================================================

  /// Save a database snapshot to [path].
  void save(String path) {
    _reserve();
    try {
      final pathPtr = path.toNativeUtf8(allocator: malloc);
      try {
        final status = _bindings.grafeoSave(_handle, pathPtr);
        if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      } finally {
        malloc.free(pathPtr);
      }
    } finally {
      _release();
    }
  }

  /// Force a WAL checkpoint.
  void walCheckpoint() {
    _reserve();
    try {
      final status = _bindings.grafeoWalCheckpoint(_handle);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      _release();
    }
  }
}
