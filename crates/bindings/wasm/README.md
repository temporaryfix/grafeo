# @grafeo-db/wasm

Low-level WebAssembly binary for [Grafeo](https://github.com/GrafeoDB/grafeo), a high-performance graph database.

## Which Package Do You Need?

| Package | Use Case |
|---------|----------|
| [`@grafeo-db/web`](https://www.npmjs.com/package/@grafeo-db/web) | Browser apps with IndexedDB, Web Workers, React/Vue/Svelte (recommended) |
| [`@grafeo-db/wasm`](https://www.npmjs.com/package/@grafeo-db/wasm) | Raw WASM binary for custom loaders or non-standard runtimes |
| [`@grafeo-db/js`](https://www.npmjs.com/package/@grafeo-db/js) | Node.js native bindings (faster than WASM for server-side) |

**Most users should use `@grafeo-db/web`** - it wraps this package and adds browser-specific features.

## Installation

```bash
npm install @grafeo-db/wasm
```

## Usage

```typescript
import init, { Database } from '@grafeo-db/wasm';

// Initialize the WASM module
await init();

// Create a database and query
const db = new Database();
const result = db.execute(`MATCH (n:Person) RETURN n.name`);
```

## Status

- [x] Core WASM bindings via wasm-bindgen
- [x] In-memory database support
- [x] GQL query language (default via `edge` profile)
- [x] TypeScript type definitions
- [x] Size-optimized build profile (absolute release size qualification remains open)
- [x] Vector search bindings (k-NN, MMR)
- [x] Snapshot export/import for IndexedDB persistence
- [x] Batch import (importLpg, importRdf, importRows)
- [x] Memory introspection (memoryUsage)

The default `edge` profile exposes direct LPG creation/import, counts, explicit
transactions, schema/info/memory inspection and unsigned/authenticated snapshots.
These methods follow compiled LPG capability in `edge`, `lpg`, `native` and
`compact-store`. Compact maintenance enables its LPG prerequisite automatically.
`native` selects parser-free, in-memory LPG + RDF for WASM; it does not enable
host filesystem or WAL backends. Add `gql` for GQL execution and streaming with
`native` or `compact-store`. Query parsers remain separately selected features.
Default query calls without GQL or Cypher report `GRAFEO-Q004` (unsupported);
explicit selection of an unavailable language retains `GRAFEO-Q002`.

`info()` is also available for RDF databases. `rdf-model` enables structured
`importRdf` and exact quad operations, including in `full` and parser-free model
builds. Create RDF databases with `Database.withGraphModel("rdf")`; SPARQL
execution additionally requires its parser feature.

The `rabitq-codec`, `fsst-codec` and `webgraph-codec` features expose standalone
compression APIs without enabling a database/query profile. RaBitQ encoding
rejects overflowing or mismatched vector dimensions with a JavaScript error.

The `opfs` feature adds browser persistence for the content-addressed block pool.
It works without selecting a database/query profile.

## Package Contents

```
@grafeo-db/wasm/
├── grafeo_wasm_bg.wasm    # WebAssembly binary
├── grafeo_wasm.js         # JavaScript loader
├── grafeo_wasm.d.ts       # TypeScript definitions
└── package.json
```

## Bundle Size

The default artifact has a 660 KiB gzipped release budget. Current qualification
has not met that budget; historical size figures no longer describe this build.
Run `scripts/build-wasm.sh` from the repository root to measure the selected
profile. AI and full profiles require their own artifact qualification.

## Runtime Support

| Runtime | Status |
|---------|--------|
| Browser (Chrome, Firefox, Safari, Edge) | Supported |
| Deno | Supported |
| Cloudflare Workers | Untested |
| Node.js | Use `@grafeo-db/js` instead |

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Roadmap](https://github.com/GrafeoDB/grafeo/blob/main/docs/roadmap.md)

## License

Apache-2.0

## Index owners

`createIndex(request)` returns an unsigned 32-bit owner ID.
The request includes `property`, optional `kind` (property/btree/text/vector),
`graph`, `name`, and `label` (text/vector only). Graphs are component arrays:
`[]` is root; `[""]` is an empty child; `["a/b"]` differs from `["a", "b"]`.
Vector-only fields are dimensions, metric, m, efConstruction, and quantization.
Text-only `minTokenLength` counts UTF-8 bytes and defaults to 2 when omitted
(or `undefined`); explicit 0 is valid. Supply a nonnegative safe integer no
greater than 4,294,967,295.
Null, coercible/non-numeric values, and use with another kind throw without
allocating an owner. Rebuild and snapshot import retain the resolved tokenizer.

```typescript
const owner = db.createIndex({ property: 'email' });
db.rebuildIndex(owner); // Atomic; preserves owner and resolved configuration.
db.dropIndex(owner);    // Returns false only when the owner is absent.
```

The `index_input` WASM runtime target includes tokenizer request validation and,
with `lpg,text-index`, token-sensitive rebuild/snapshot controls. With Text
disabled it checks structured errors without consuming an owner. Run it with the
installed `wasm-bindgen-test-runner`, for example from the repository root:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test --locked -p grafeo-wasm --target wasm32-unknown-unknown --no-default-features --features lpg,text-index --test index_input tokenizer
```

Duplicate creation and missing-owner rebuild throw; explicit recreation
returns a new owner. Invalid paths/options and unavailable features report errors.

## Bounded queries and cancellation

`executeWithOptions` and `executeRawWithOptions` accept a single-use
`QueryControl`, output limits, and parameters. Eager output defaults to
1,000,000 rows and 64 MiB. Copy admission happens before a mutation commits;
a rejected statement leaves earlier transaction writes intact.

```typescript
import { QueryControl } from '@grafeo-db/wasm';

const control = new QueryControl();
const stream = db.executeStreamWithOptions(
  'MATCH (n:Person) RETURN n.name', control,
  { maxBytes: 1024 * 1024 }, undefined,
);
try {
  for (;;) {
    const chunk = stream.nextChunk(256);
    if (chunk === null) break;
    console.log(chunk);
  }
} finally {
  try { stream.close(); } finally { stream.free(); control.free(); }
}
```

`next()` returns one row, `nextChunk(n)` returns at most `min(n, 1024)` rows,
and `toArray()` collects within the total row/byte limits. Chunk iteration bounds
each copied chunk; it does not impose the eager default total row cap unless
`maxRows` is supplied. Close streams explicitly on early exit. Independent read
streams may coexist; other database operations report busy until they close.
A stream retains its database even if the JavaScript database wrapper is freed.

Call `control.cancel()` between pulls to cancel a stream. Cancellation and other
terminal failures remain observable from subsequent pulls and `close()`, with
native error identity in `Error.code`. A synchronous call blocks its own event
loop, so that loop cannot interrupt it. Explicit `QueryControl(timeoutMs)`
deadlines currently throw `GRAFEO-Q004`; omitted deadlines remain cancellable.
Streaming within an explicit transaction, or without the required GQL/LPG
features, also returns structured unsupported errors.

## Bounded change pages (`cdc` feature)

Build with `--features cdc` (also included in `full`) to expose `setCdcEnabled`,
`isCdcEnabled`, `changesAfter`, `nodeHistoryAfter` and `edgeHistoryAfter`. Default
features are unchanged. Enable capture before creating the sessions/transactions
whose changes you need. Pages contain owned JS events and a 97-byte `Uint8Array`
exclusive cursor; no page handle needs freeing, including on early stop.

```js
db.setCdcEnabled(true);
db.execute('INSERT (:Event)');
const page = db.changesAfter(null, 100, 64 * 1024);
const tail = db.changesAfter(page.next, 100, 64 * 1024);
const history = db.nodeHistoryAfter(page.events[0].entity_id, '0', null, 100, 64 * 1024);
```

IDs,epochs,timestamps,incarnations and edge endpoints are exact decimal strings.
Entity selectors and inclusive epochs accept decimal u64 strings. Required limits
are positive integers fitting WASM size bounds; bytes count native event encodings
without JS/page envelopes. Invalid/foreign/evicted/resource errors retain native
`Error.code`. Empty filtered pages may advance, so EOF means an unchanged cursor.

Reads use the existing active transaction Session and never expose uncommitted
events. In-memory state lives only for the process. With `lpg,cdc`, the existing
`exportSnapshot`/`importSnapshot` path preserves the feed and cursor identity;
the caller must store that snapshot durably. CDC does not add browser persistence.
