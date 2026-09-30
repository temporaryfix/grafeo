# Grafeo Dart Bindings

Dart FFI bindings for the [Grafeo](https://grafeo.dev) graph database. Wraps the `grafeo-c` shared library for native performance with a Dart-idiomatic API.

## Installation

```yaml
dependencies:
  grafeo: ^0.0.1
```

You also need the `grafeo-c` native library for your platform. See [Building from Source](#building-from-source) below.

## Quick Start

```dart
import 'package:grafeo/grafeo.dart';

void main() {
  final db = GrafeoDB.memory();

  // Insert data
  db.execute("INSERT (:Person {name: 'Alix', age: 30})");
  db.execute("INSERT (:Person {name: 'Gus', age: 28})");

  // Query with parameters
  final result = db.executeWithParams(
    r'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name, p.age',
    {'minAge': 25},
  );

  for (final row in result.rows) {
    print('${row['p.name']}: ${row['p.age']}');
  }

  // Transactions
  final tx = db.beginTransaction();
  tx.execute("INSERT (:City {name: 'Amsterdam'})");
  tx.execute("INSERT (:City {name: 'Berlin'})");
  tx.commit();

  // CRUD operations
  final nodeId = db.createNode(['Person'], {'name': 'Vincent'});
  db.setNodeProperty(nodeId, 'role', 'hitman');
  final node = db.getNode(nodeId);
  print(node); // Node(id, [Person], {name: Vincent, role: hitman})

  db.close();
}
```

## Bounded queries and cancellation

```dart
final control = QueryControl(timeout: const Duration(seconds: 5));
try {
  final pending = db.executeWithOptionsAsync(
    'MATCH (p:Person) RETURN p.name',
    options: ExecutionOptions(control: control, maxRows: 10000, maxBytes: 16 * 1024 * 1024),
  );
  // control.cancel() can interrupt native work from this isolate.
  final result = await pending;
  print(result.rows.length);
} finally {
  control.close();
}

final cursor = db.executeStreamWithOptions('MATCH (p:Person) RETURN p.name');
try {
  await for (final row in cursor.rowsAsync()) {
    print(row);
  }
} finally {
  await cursor.closeAsync();
}
```

Controls are single-use, with deadlines starting at construction. Async query
and pull methods run native work on worker isolates and retain their owners until
those workers exit. Synchronous calls block the calling isolate. Use `closeAsync`
to cancel and join an active cursor pull; synchronous `close` reports busy.
Closing a busy database preserves its handle so you can finish work and retry.

`GrafeoException.code` preserves `GRAFEO-Q007` cancellation, `GRAFEO-Q003`
deadlines and `GRAFEO-S001` limits. Failed statements preserve earlier transaction
writes. Eager results and `toList` default to one million rows and a 64 MiB copy
envelope; streams apply byte limits per row/chunk and optional total row caps.
`nextChunk` returns at most 1024 rows. Collection failures return no partial list.
Terminal errors remain observable through subsequent pulls and close. Async
iterator cancellation closes the cursor; synchronous `rows()` requires explicit
close after early break because Dart iterators have no disposal callback.

Temporal marker maps with unexpected types or timestamps outside Dart's date
range remain exact maps. Supported timestamps decode to UTC `DateTime`. Result
copy limits do not include engine execution memory or application-retained rows.

## API Reference

### GrafeoDB

| Method | Description |
|--------|-------------|
| `GrafeoDB.memory()` | Open an in-memory database |
| `GrafeoDB.open(path)` | Open a persistent database (directory or single-file) |
| `GrafeoDB.openSingleFile(path)` | Open a single-file `.grafeo` database |
| `GrafeoDB.openReadOnly(path)` | Open an existing database in read-only mode |
| `GrafeoDB.version()` | Get the library version string |
| `execute(query)` | Execute a GQL query |
| `executeWithParams(query, params)` | Execute GQL with parameters |
| `executeCypher(query)` | Execute a Cypher query |
| `executeCypherWithParams(query, params)` | Execute Cypher with parameters |
| `executeGremlin(query)` | Execute a Gremlin query |
| `executeGremlinWithParams(query, params)` | Execute Gremlin with parameters |
| `executeGraphql(query)` | Execute a GraphQL query |
| `executeGraphqlWithParams(query, params)` | Execute GraphQL with parameters |
| `executeSparql(query)` | Execute a SPARQL query |
| `executeSparqlWithParams(query, params)` | Execute SPARQL with parameters |
| `executeLanguage(lang, query, {params})` | Execute in any supported language |
| `setSchema(name)` | Set active schema for subsequent queries |
| `resetSchema()` | Revert to default graph store |
| `currentSchema()` | Get active schema name (or null) |
| `beginTransaction()` | Start an ACID transaction |
| `beginTransactionWithIsolation(level)` | Start transaction with isolation level |
| `createNode(labels, properties)` | Create a node, returns ID |
| `getNode(id)` | Get a node by ID |
| `getNodeLabels(id)` | Get labels only (faster than getNode) |
| `deleteNode(id)` | Delete a node |
| `createEdge(src, dst, type, props)` | Create an edge, returns ID |
| `getEdge(id)` | Get an edge by ID |
| `deleteEdge(id)` | Delete an edge |
| `setNodeProperty(id, key, value)` | Set a node property |
| `setEdgeProperty(id, key, value)` | Set an edge property |
| `removeNodeProperty(id, key)` | Remove a node property |
| `removeEdgeProperty(id, key)` | Remove an edge property |
| `addNodeLabel(id, label)` | Add a label to a node |
| `removeNodeLabel(id, label)` | Remove a label from a node |
| `createIndex(request)` | Create a Property/BTree/Text/Vector catalog owner |
| `dropIndex(owner)` | Drop an owner; false only when absent |
| `rebuildIndex(owner)` | Rebuild an owner; missing owners throw |
| `hasPropertyIndex(key)` | Check if property index exists |
| `findNodesByProperty(key, value)` | Find node IDs by indexed property |
| `vectorSearch(label, prop, query, {k, ef})` | k-NN vector search |
| `mmrSearch(label, prop, query, {k, ...})` | MMR diversity-aware search |
| `batchCreateNodes(label, prop, vectors)` | Bulk-create nodes with embeddings |
| `nodeCount` | Number of nodes |
| `edgeCount` | Number of edges |
| `info()` | Database metadata as JSON map |
| `save(path)` | Save snapshot to path |
| `walCheckpoint()` | Force WAL checkpoint |
| `close()` | Close and flush |

```dart
final owner = db.createIndex(const CreateIndexRequest(
  kind: IndexKind.vector, label: 'Doc', property: 'embedding', dimensions: 3,
));
db.rebuildIndex(owner);
final dropped = db.dropIndex(owner);
```

An empty `graph` list selects root; each string is one literal component,
including empty names, separators and NULs. Null options are absent, while
explicit zero/empty values reach engine validation unchanged. Property/BTree
omit `label`; Text/Vector require it. Missing rebuild and all engine failures
throw exceptions; drop returns false only for absence. Features and reads remain.

### Transaction

| Method | Description |
|--------|-------------|
| `execute(query)` | Execute GQL within transaction |
| `executeWithParams(query, params)` | Execute GQL with parameters |
| `executeLanguage(lang, query, {params})` | Execute in any language |
| `commit()` | Make changes permanent |
| `rollback()` | Discard changes |

### Types

- **`QueryResult`**: rows, columns, nodes, edges, executionTimeMs, rowsScanned
- **`Node`**: id, labels, properties
- **`Edge`**: id, type, sourceId, targetId, properties
- **`VectorResult`**: nodeId, distance

## Building from Source

```bash
# Clone and build the native library
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
cargo build --release -p grafeo-c

# Copy to the Dart package (or your project)
# Linux:   cp target/release/libgrafeo_c.so crates/bindings/dart/
# macOS:   cp target/release/libgrafeo_c.dylib crates/bindings/dart/
# Windows: copy target\release\grafeo_c.dll crates\bindings\dart\

# Run tests
cd crates/bindings/dart
dart pub get
dart test
```

## License

Apache-2.0. See [LICENSE](../../LICENSE) for details.

## Bounded change pages

Enable `db.cdcEnabled`, then call `changesAfter(cursor, maxEvents: ..., maxBytes: ...)`.
Pass `null` initially; resume with `page.next`. `nodeHistoryAfter` and
`edgeHistoryAfter` take a `BigInt` entity ID and optional inclusive `sinceEpoch`.
Coordinates use `BigInt` to preserve native unsigned 64-bit values; use
`BigInt.from(id)` for an ordinary Dart CRUD ID. Creation events include labels,
edge type/endpoints and graph identity. Page data survives `db.close()` and
requires no cleanup; each native page is freed before the call returns.

```dart
final page = db.changesAfter(null, maxEvents: 100, maxBytes: 64 * 1024);
final next = db.changesAfter(page.next, maxEvents: 100, maxBytes: 64 * 1024);
```

Both bounds must be positive. The byte bound counts native event encodings,
excluding JSON/page envelopes. Cursors are opaque canonical 97-byte values;
non-null empty values are invalid. Empty filtered pages can advance, so EOF is
an unchanged cursor, not simply an empty event list. Native structured errors
retain their codes, including invalid/foreign cursor, evicted cursor and resource
limits. These APIs require a native library built with CDC support.
