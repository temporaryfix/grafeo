# @grafeo-db/js

Node.js/TypeScript bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Installation

Requires Node.js 20.3.0 or newer (Node-API 9).

```bash
npm install @grafeo-db/js
```

## Quick Start

```typescript
import { GrafeoDB } from '@grafeo-db/js';

// In-memory database
const db = GrafeoDB.create();

// Or persistent
// const db = GrafeoDB.create('./my-graph');

// Create nodes
db.createNode(['Person'], { name: 'Alix', age: 30 });
db.createNode(['Person'], { name: 'Gus', age: 25 });
db.createEdge(0, 1, 'KNOWS', { since: 2024 });

// Query with GQL
const result = await db.execute('MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age');
for (const row of result.toArray()) {
  console.log(row);
}

db.close();
```

## API Reference

### Database

```typescript
// Create / open
const db = GrafeoDB.create();           // in-memory
const db = GrafeoDB.create('./path');    // persistent
const db = GrafeoDB.open('./path');      // open existing

// Counts
db.nodeCount();   // number of nodes
db.edgeCount();   // number of edges
```

### Query Languages

All query methods return `Promise<QueryResult>` and accept optional parameters and execution options:

```typescript
import { QueryControl } from '@grafeo-db/js';

const control = new QueryControl(10_000); // deadline starts at construction
await db.execute(gql, params, { control, maxRows: 1_000_000, maxBytes: 64 * 1024 * 1024 });
await db.executeCypher(query, params, options);  // Cypher
await db.executeGremlin(query, params, options); // Gremlin
await db.executeGraphql(query, params, options); // GraphQL
await db.executeSparql(query, params, options);  // SPARQL
await db.executeSql(query, params, options);     // SQL/PGQ (SQL:2023)
// Call control.cancel() while the query is pending to request cancellation.
```

Native eager results default to 1,000,000 rows and 64 MiB of retained result
storage. `maxRows` and `maxBytes` override those limits; copied binding output
uses the same byte cap with conservative conversion accounting. A deadline reports `GRAFEO-Q003` and a result limit
reports `GRAFEO-S001`; native failures expose the identifier as `error.code`.
A control is consumed synchronously and cannot be reused. Limits must be
nonnegative JavaScript safe integers; zero admits no rows or bytes.

### Streaming

```typescript
const stream = await db.executeStream(gql, params, options);
for await (const row of stream) {
  console.log(row);
  break; // awaits native close() through the async iterator return() hook
}
await stream.close(); // idempotent; next() then resolves to null
```

Streams retain bounded native state and charge each copied row against
`maxBytes`. An explicit `maxRows` also bounds the total emitted rows; without
it, the stream has no total row cap. `close()` releases the cursor asynchronously
and interrupts an active pull. Native cancellation or cleanup errors remain
observable from close; repeated calls preserve that outcome. `next()` remains
available and resolves to `null` at EOF or after close. Database `close()`
reports `GRAFEO-T004` promptly while a query, stream or transaction is active;
finish or close those owners before retrying.

### Node & Edge CRUD

```typescript
const node = db.createNode(['Label'], { key: 'value' });
const edge = db.createEdge(sourceId, targetId, 'TYPE', { key: 'value' });

const n = db.getNode(id);     // JsNode | null
const e = db.getEdge(id);     // JsEdge | null

db.setNodeProperty(id, 'key', 'value');
db.setEdgeProperty(id, 'key', 'value');

db.deleteNode(id);  // returns boolean
db.deleteEdge(id);  // returns boolean
```

### Transactions

```typescript
const tx = db.beginTransaction();
try {
  await tx.execute("INSERT (:Person {name: 'Harm'})");
  tx.commit();
} catch (e) {
  tx.rollback();
}

// Node.js 22+ with explicit resource management:
using tx = db.beginTransaction();
await tx.execute("INSERT (:Person {name: 'Harm'})");
tx.commit(); // auto-rollback if not committed
```

### QueryResult

```typescript
result.columns;          // column names
result.length;           // row count
result.executionTimeMs;  // execution time (ms)
result.get(0);           // single row as object
result.toArray();        // all rows as objects
result.scalar();         // first column of first row
result.nodes();          // extracted nodes
result.edges();          // extracted edges
```

### Graph Projections

```typescript
db.createProjection('people', ['Person'], ['KNOWS']); // returns boolean
const projections = db.listProjections();  // ['people']
db.dropProjection('people');
```

### Data Import

```typescript
const count = await db.importCsv('./users.csv', { label: 'Person', headers: true });
const count2 = await db.importJsonl('./events.jsonl', { label: 'Event' });
```

### Backup and Restore

```typescript
db.backupFull('/backups/full');
db.backupIncremental('/backups/incr');
GrafeoDB.restoreToEpoch('/backups/full', 100, './restored');
```

### Vector Search

```typescript
// Create an HNSW index
await db.createIndex({ kind: 'vector', label: 'Document', property: 'embedding', dimensions: 384 });

// Bulk insert
const ids = await db.batchCreateNodes('Document', 'embedding', vectors);

// Search
const results = await db.vectorSearch('Document', 'embedding', queryVector, 10);
```

## Features

- GQL, Cypher, SPARQL, Gremlin, GraphQL and SQL/PGQ query languages
- Full node/edge CRUD with property management
- ACID transactions with automatic rollback
- HNSW vector similarity search with batch operations
- Graph projections (filtered virtual views)
- CSV and JSON Lines import
- Incremental backup and restore
- Async/await API backed by Rust + Tokio
- TypeScript definitions included

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Python Package](https://pypi.org/project/grafeo/)
- [WASM Package](https://www.npmjs.com/package/@grafeo-db/wasm)

## License

Apache-2.0

## Index owners

`createIndex(request)` resolves to an unsigned 32-bit owner ID.
The request includes `property`, optional `kind` (property/btree/text/vector),
`graph`, `name`, and `label` (text/vector only). Graphs are component arrays:
`[]` is root; `[""]` is an empty child; `["a/b"]` differs from `["a", "b"]`.
Vector-only fields are dimensions, metric, m, efConstruction, and quantization.
Text-only `minTokenLength` counts UTF-8 bytes and defaults to 2 when omitted
(or `undefined`); explicit 0 is valid. Supply a nonnegative safe integer fitting
the platform size range.
Null, coercible/non-numeric values, and use with another kind reject without
allocating an owner. Rebuild and persistence retain the resolved tokenizer.
Unknown keys (including non-enumerable or symbol keys), inherited enumerable
options, and malformed UTF-16 strings reject the Promise before mutation.
Valid Unicode is preserved exactly, including graph path components.

```typescript
const owner = await db.createIndex({ property: 'email' });
await db.rebuildIndex(owner); // Atomic; preserves owner and resolved configuration.
await db.dropIndex(owner);    // Resolves to false only when the owner is absent.
```

The standalone tokenizer controls need no npm dependencies. From the repository
root, point them at the newly built local addon with GQL, Text and storage enabled:

```sh
GRAFEO_NODE_LIBRARY=/absolute/path/to/libgrafeo_node.dylib node --test crates/bindings/node/__test__/index-tokenizer.node.mjs
```

Against a local LPG addon built without Text support, set
`GRAFEO_EXPECT_TEXT_DISABLED=1` for the same command. That mode checks structured
feature errors and owner-allocation safety instead of skipping unavailable behavior.

Duplicate creation and missing-owner rebuild reject; explicit recreation
returns a new owner. Invalid paths/options and unavailable features report errors.

### Building the Node package

Release tools are pinned separately from the optional native packages. From
`crates/bindings/node`, install them with `npm ci --prefix tools --ignore-scripts`.
The Linux GNU producer uses Node.js 24.21.0, Rust 1.97.1 and
`bash scripts/build-node-linux.sh x86_64-unknown-linux-gnu` from the repository
root (use `aarch64-unknown-linux-gnu` on ARM64). It writes the addon and generated
loader/declarations under `target/node-release/package/generated`. Package these
outputs together; include the root Apache license in the main and native packages.
