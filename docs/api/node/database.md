---
title: GrafeoDB
description: GrafeoDB class reference for Node.js.
tags:
  - api
  - nodejs
---

# GrafeoDB

The main database class. All query methods return `Promise<QueryResult>`.

## Constructor

```typescript
// In-memory database
const db = GrafeoDB.create();

// Persistent database
const db = GrafeoDB.create('./my_graph.db');

// Select RDF-only or dual-model storage at creation time
const rdfDb = GrafeoDB.create(undefined, 'rdf');
const mixedDb = GrafeoDB.create('./mixed.db', 'both');

// Open existing database
const db = GrafeoDB.open('./my_graph.db');
```

### Parameters

| Method | Parameters | Description |
|--------|-----------|-------------|
| `create(path?, graphModel?)` | `path: string \| null \| undefined`, `graphModel: "lpg" \| "rdf" \| "both"` | Create a database (in-memory if no path); LPG is the default model |
| `open(path)` | `path: string` | Open an existing database |

The graph model is persisted and fixed when the database is created. Opening a
database recovers its stored model; query methods do not silently add a second
model.

The `lpg`, `embedded`, `edge`, `native` and `full` Node profiles expose native
node/edge CRUD and transaction `createNode`. The `native` profile keeps query
parsers disabled. The memory-only `edge` profile omits save/backup methods;
adding storage enables persistence. Saving works with either stored graph model;
backup methods additionally require compiled LPG support.

## Native RDF insertion

With `triple-store` support, RDF and dual-model databases accept native quads
without a query parser. Terms use N-Triples spelling or bare IRIs. Each bulk
item is `[subject, predicate, object]` or includes a fourth graph IRI.

```typescript
const db = GrafeoDB.create(undefined, 'rdf');
const [inserted, epoch] = db.insertRdfQuads([
  ['http://example.org/s', 'http://example.org/p', '"value"'],
]);
// inserted is a number; epoch is an exact unsigned decimal string.
const exactEpoch = BigInt(epoch);
db.close();
```

`insertRdfQuad(subject, predicate, object, graph?)` returns numeric `0` or `1`.
`insertRdfQuads(quads)` returns `[insertedCount, epochString]`. Counts fit an
unsigned 32-bit integer; oversized bulk input is rejected before mutation.
The epoch remains exact above JavaScript's safe-integer and signed 64-bit ranges.
Transactions expose the same two insertion methods, both returning numeric
counts; changes become visible outside the transaction after commit. Duplicate
quads do not increase the inserted count. `containsRdfQuad(...)` checks exact
typed membership in the database or transaction.

## Query Methods

All query methods accept an optional `params` object for parameterized queries.

### execute()

Execute a GQL (ISO standard) query.

```typescript
async execute(query: string, params?: object): Promise<QueryResult>
```

```typescript
const result = await db.execute(
  'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
  { minAge: 25 }
);
```

### executeCypher()

Execute a Cypher query. Requires the `cypher` feature.

```typescript
async executeCypher(query: string, params?: object): Promise<QueryResult>
```

### executeGremlin()

Execute a Gremlin query. Requires the `gremlin` feature.

```typescript
async executeGremlin(query: string, params?: object): Promise<QueryResult>
```

### executeGraphql()

Execute a GraphQL query. Requires the `graphql` feature.

```typescript
async executeGraphql(query: string, params?: object): Promise<QueryResult>
```

### executeSparql()

Execute a SPARQL query against the RDF triple store. Requires the `sparql` feature.

```typescript
async executeSparql(query: string, params?: object): Promise<QueryResult>
```

The database must have been created with `graphModel` set to `"rdf"` or
`"both"`.

### executeSql()

Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE). Requires the `sql-pgq` feature.

```typescript
async executeSql(query: string, params?: object): Promise<QueryResult>
```

```typescript
const result = await db.executeSql(
  'SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))'
);
```

## Node Operations

### createNode()

Create a node with labels and optional properties.

```typescript
createNode(labels: string[], properties?: object): JsNode
```

```typescript
const node = db.createNode(['Person'], { name: 'Alix', age: 30 });
console.log(node.id);     // 0
console.log(node.labels);  // ['Person']
```

### getNode()

Get a node by ID. Returns `null` if not found.

```typescript
getNode(id: number): JsNode | null
```

### deleteNode()

Delete a node by ID. Returns `true` if the node existed.

```typescript
deleteNode(id: number): boolean
```

### setNodeProperty()

Set a property on a node.

```typescript
setNodeProperty(id: number, key: string, value: any): void
```

Returns normally only after a successful write. A missing or deleted node throws
an `Error`; the setter does not create the node. Property-size/schema rejection,
closed-database and durability failures also propagate as errors.

### removeNodeProperty()

Remove a property from a node. Returns `true` if the property existed.

```typescript
removeNodeProperty(id: number, key: string): boolean
```

### addNodeLabel()

Add a label to an existing node. Returns `true` if the label was added.

```typescript
addNodeLabel(id: number, label: string): boolean
```

### removeNodeLabel()

Remove a label from a node. Returns `true` if the label was removed.

```typescript
removeNodeLabel(id: number, label: string): boolean
```

### getNodeLabels()

Get all labels for a node. Returns `null` if the node doesn't exist.

```typescript
getNodeLabels(id: number): string[] | null
```

## Edge Operations

### createEdge()

Create an edge between two nodes with a type and optional properties.

```typescript
createEdge(sourceId: number, targetId: number, edgeType: string, properties?: object): JsEdge
```

```typescript
const edge = db.createEdge(0, 1, 'KNOWS', { since: 2024 });
console.log(edge.edgeType);  // 'KNOWS'
console.log(edge.sourceId);  // 0
console.log(edge.targetId);  // 1
```

### getEdge()

Get an edge by ID. Returns `null` if not found.

```typescript
getEdge(id: number): JsEdge | null
```

### deleteEdge()

Delete an edge by ID. Returns `true` if the edge existed.

```typescript
deleteEdge(id: number): boolean
```

### setEdgeProperty()

Set a property on an edge.

```typescript
setEdgeProperty(id: number, key: string, value: any): void
```

Returns normally only after a successful write. A missing or deleted edge throws
an `Error`; the setter does not create the edge. Other engine write failures
also propagate as errors.

### removeEdgeProperty()

Remove a property from an edge. Returns `true` if the property existed.

```typescript
removeEdgeProperty(id: number, key: string): boolean
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodeCount` | `number` | Number of nodes in the database |
| `edgeCount` | `number` | Number of edges in the database |

## Transaction Methods

### beginTransaction()

Start a new transaction with an optional isolation level.

```typescript
beginTransaction(isolationLevel?: string): Transaction
```

Isolation levels: `"read_committed"`, `"snapshot"` (default), `"serializable"`.

```typescript
const tx = db.beginTransaction();
const tx = db.beginTransaction('serializable');
```

## Index Owners

```typescript
interface CreateIndexRequest {
  property: string;
  kind?: "property" | "btree" | "text" | "vector";
  graph?: string[];
  name?: string;
  label?: string;
  dimensions?: number;
  metric?: string;
  m?: number;
  efConstruction?: number;
  quantization?: string;
}
createIndex(request: CreateIndexRequest): Promise<number>
dropIndex(owner: number): Promise<boolean>
rebuildIndex(owner: number): Promise<void>
```

Creation returns a committed unsigned 32-bit owner ID. Duplicate names or physical targets are errors. Graph paths are component arrays: `[]` selects root, `[""]` an empty-named child, and `["a/b"]` differs from `["a", "b"]`. Property/BTree indexes forbid a label; Text/Vector require one. Rebuild atomically preserves the owner and its full resolved configuration; it does not recreate a dropped index. Drop returns false only for an absent owner. Engine failures propagate through the binding's error channel.

All three methods are asynchronous. Omitted `kind` means `"property"`.
The numeric/configuration options apply only to Vector; invalid options and
unavailable features reject the promise. Normal writes maintain indexes
automatically.

```typescript
const vectorOwner = await db.createIndex({
  kind: "vector", label: "Document", property: "embedding",
  dimensions: 384, metric: "cosine", quantization: "scalar",
});
await db.rebuildIndex(vectorOwner);
const dropped = await db.dropIndex(vectorOwner);
```

Current 0.0.1 limitation: index-owner mutations on WAL-backed databases are rejected. Saving or checkpointing owner-bearing state, including retained owner-ID allocation history after drops, also fails closed until the current persistence formats support those owners. The in-memory examples below are not a persistence guarantee.

## Vector Search

### vectorSearch()

Search for the k nearest neighbors of a query vector.
Returns `[[nodeId, distance], ...]` sorted by distance ascending (lower distance = more similar). The distance scale depends on the metric configured at index creation: cosine `[0, 2]`, euclidean `[0, inf)`, dot_product (negated), manhattan `[0, inf)`.

```typescript
async vectorSearch(
  label: string,
  property: string,
  query: number[],
  k: number,
  ef?: number,
  filters?: Record<string, any>
): Promise<number[][]>
```

```typescript
const results = await db.vectorSearch('Document', 'embedding', queryVec, 10);
for (const [nodeId, distance] of results) {
  console.log(`Node ${nodeId}: distance ${distance}`);
}

// With metadata filter
const filtered = await db.vectorSearch(
  'Document', 'embedding', queryVec, 10, undefined, { user_id: 1 }
);
```

### batchCreateNodes()

Bulk-insert nodes with vector properties. Returns an array of node IDs.

```typescript
async batchCreateNodes(
  label: string,
  property: string,
  vectors: number[][]
): Promise<number[]>
```

### batchVectorSearch()

Batch search for nearest neighbors of multiple query vectors.

```typescript
async batchVectorSearch(
  label: string,
  property: string,
  queries: number[][],
  k: number,
  ef?: number,
  filters?: Record<string, any>
): Promise<number[][][]>
```

### mmrSearch()

Search for diverse nearest neighbors using Maximal Marginal Relevance. Returns `[[nodeId, distance], ...]` in MMR selection order. The `distance` values are identical to those returned by `vectorSearch()` for the same nodes (lower = more similar). The ordering reflects MMR's relevance-diversity balance, not pure distance sorting.

```typescript
async mmrSearch(
  label: string,
  property: string,
  query: number[],
  k: number,
  fetchK?: number,
  lambdaMult?: number,  // diversity vs relevance (0 = max diversity, 1 = max relevance)
  ef?: number,
  filters?: Record<string, any>
): Promise<number[][]>
```

## Text Search

Create Text indexes with `await db.createIndex({kind: "text", label: "Article", property: "title"})`. They are maintained automatically by normal writes.

### textSearch()

Search a text index using BM25 scoring. Returns `[[nodeId, score], ...]` sorted by descending relevance (higher score = more relevant). BM25 scores are unbounded positive floats.

```typescript
async textSearch(
  label: string,
  property: string,
  query: string,
  k: number
): Promise<number[][]>
```

### hybridSearch()

Combine text (BM25) and vector similarity search. For best results, create both a text index (`createIndex({kind: "text", ...})`) and a vector index (`createIndex({kind: "vector", ...})`). If either index is missing, that source is silently omitted from fusion.

Returns `[[nodeId, score], ...]` sorted by fused score **descending** (higher = more relevant). These are fusion scores, **not** distances.

!!! warning "Score convention differs from vectorSearch"
    `hybridSearch()` returns fusion scores where higher = better.
    `vectorSearch()` returns distances where lower = better.
    For temporal decay, **multiply** fusion scores but **divide** distances.

```typescript
async hybridSearch(
  label: string,
  textProperty: string,
  vectorProperty: string,
  queryText: string,
  k: number,
  queryVector?: number[],
  fusion?: string,         // 'weighted' for weighted fusion
  weights?: number[]       // [textWeight, vectorWeight], default [0.5, 0.5]
): Promise<number[][]>
```

## Embedding (opt-in)

These methods require the `embed` feature flag.

### registerEmbeddingModel()

Register an ONNX embedding model for text-to-vector conversion.

```typescript
async registerEmbeddingModel(
  name: string,
  modelPath: string,
  tokenizerPath: string,
  batchSize?: number
): Promise<void>
```

### embedText()

Generate embeddings for a list of texts. Returns one float array per input text.

```typescript
async embedText(modelName: string, texts: string[]): Promise<number[][]>
```

### vectorSearchText()

Search a vector index using a text query, generating the embedding on-the-fly.

```typescript
async vectorSearchText(
  label: string,
  property: string,
  modelName: string,
  queryText: string,
  k: number,
  ef?: number
): Promise<number[][]>
```

## Change Data Capture

These methods require the `cdc` feature flag.

### nodeHistoryAfter() / edgeHistoryAfter()

Read owned bounded entity history through the shared durable feed. IDs and the
optional inclusive epoch are exact unsigned decimal strings. Both limits are
required positive safe integers. Pass `null` initially, then the exact `next`
Buffer; stop only when it is unchanged, even for an empty filtered page.

```typescript
nodeHistoryAfter(nodeId: string, cursor: Buffer | null, maxEvents: number,
  maxBytes: number, sinceEpoch?: string): Promise<JsChangePage>
edgeHistoryAfter(edgeId: string, cursor: Buffer | null, maxEvents: number,
  maxBytes: number, sinceEpoch?: string): Promise<JsChangePage>
```

The entity index skips unrelated entities. Graph-local ID collisions remain
aggregated across authorized coordinates and incarnations. Returned event
coordinates use exact decimal strings and native errors retain `error.code`.

### changesAfter()

Returns an owned bounded page with canonical cursor bytes. Both limits are
positive safe integers. Pass `null` initially, then return the exact `next`
Buffer. Stop only when cursor bytes are unchanged, even for an empty page.

```typescript
changesAfter(cursor: Buffer | null, maxEvents: number, maxBytes: number): Promise<JsChangePage>
// JsChangePage: { events: [...], next: Buffer }
```

Page event `entity_id`, `epoch`, `timestamp` and non-null `graph_incarnation`
are exact decimal strings; use `BigInt` for arithmetic. Native failures retain
stable `error.code`. Pages own their data across close; reads after close fail.
See [bounded CDC pages](../../user-guide/cdc.md#bounded-feed-pages).

Whole-feed and entity pages share this event format:

Each `ChangeEvent` is a JSON object:

```typescript
{
  entity_id: string;
  entity_type: 'node' | 'edge' | 'triple';
  kind: 'create' | 'update' | 'delete';
  epoch: string;
  timestamp: string;
  graph_incarnation: string | null;
  lpg_graph: string[] | null; // [] is LPG root; RDF is null
  triple_graph: string | null; // RDF named graph only
  triple_subject: string | null;
  triple_predicate: string | null;
  triple_object: string | null;
  before: Record<string, any> | null;
  after: Record<string, any> | null;
}
```

## Admin Methods

### info()

Returns high-level database information as a JSON object.

```typescript
info(): object
```

### schema()

Returns schema information (labels, edge types, property keys).

```typescript
schema(): object
```

### version()

Returns the Grafeo engine version string.

```typescript
version(): string
```

### compact()

Converts the database to a read-only [CompactStore](../../user-guide/compact-store.md). Takes a snapshot of all nodes and edges, builds a columnar store with CSR adjacency, and switches to read-only mode. Write operations will throw after this call.

```typescript
compact(): void
```

```typescript
const db = GrafeoDB.create();
await db.execute("INSERT (:Person {name: 'Alix', age: 30})");

db.compact();

const result = await db.execute("MATCH (p:Person) RETURN p.name"); // fast
```

### close()

Close the database and release resources.

```typescript
close(): void
```
