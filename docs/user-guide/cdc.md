# Change Data Capture (CDC)

Change Data Capture records ordered node, edge and RDF triple mutations. The
opt-in native feed publishes each transaction's events with its native state.
WAL-backed databases recover the retained feed after restart; current containers,
snapshots and exact backups preserve its cursor identity and retained window.
In-memory databases keep events for their lifetime unless explicitly snapshotted.

Read the native feed through bounded `changes_after` or `history_after` pages.
Save the returned cursor to resume exclusively after the delivered position.
Retention can expire that position: `CursorEvicted` requires an application
resynchronization decision. The feed is not permanent audit history and does not
provide end-to-end exactly-once delivery to an external consumer.

The RDF dataset's `rdf_cdc_page` API additionally reads authoritative typed-Quad
interval history, including named-graph create/drop/incarnation transitions.
Its statement-history cursor is separate from the opt-in native event-feed
cursor described below; the two cursor formats are not interchangeable.

## Durable RDF dataset CDC

RDF CDC pages transitions in a fixed epoch range `(from, through]`:

```rust
let mut after = None;
loop {
    let page = db.rdf_cdc_page(from, through, after.as_ref(), 1_000)?;
    consume(&page.transitions)?;
    after = page.next_cursor;
    if !page.has_more {
        break;
    }
}
```

Keep `from` and `through` fixed while paging. Preserve `next_cursor` after the
bounded page is exhausted; the same cursor can later continue with a larger
`through`. Cursors integrity-bind the logical StoreId, graph incarnation, full
statement handle, transition kind, and ordering coordinates, so forged and
cross-store cursors fail closed. Within one commit the canonical order is:
statement retract, graph drop, graph create, statement assert.

## Enabling CDC

CDC is **opt-in** to avoid overhead on the mutation hot path.
There are two levels of control:

1. **Database-wide default**: set at construction time or toggled at runtime.
2. **Per-session override**: each session can opt in or out regardless of the
   database default.

### Rust

```rust
use grafeo_engine::{Config, GrafeoDB};

// Enable CDC for all sessions via config
let db = GrafeoDB::with_config(Config::in_memory().with_cdc())?;

// Or toggle at runtime (affects future sessions only)
db.set_cdc_enabled(true);

// Per-session override
let tracked   = db.session_with_cdc(true);   // CDC on for this session
let untracked = db.session_with_cdc(false);  // CDC off for this session
let default   = db.session();                // follows database default
```

### Python

```python
from grafeo import GrafeoDB

# Enable at construction
db = GrafeoDB(cdc=True)

# Or toggle at runtime
db.enable_cdc()
db.disable_cdc()

# Check current state
print(db.cdc_enabled)  # True / False
```

### Node.js / TypeScript

```typescript
import { GrafeoDB } from '@grafeo-db/node';

const db = GrafeoDB.create();
db.enableCdc();
console.log(db.isCdcEnabled); // true
db.disableCdc();
```

### C

```c
#include "grafeo.h"

GrafeoDatabase* db = grafeo_open_memory();
grafeo_set_cdc_enabled(db, true);
bool enabled = grafeo_is_cdc_enabled(db);
```

## Querying change history

Once CDC is enabled, every mutation records a `ChangeEvent` with:

| Field       | Description                                   |
|-------------|-----------------------------------------------|
| `entity_id` | Node/edge ID or RDF triple content hash       |
| `kind`       | `Create`, `Update`, or `Delete`              |
| `epoch`      | Commit epoch (monotonically increasing)      |
| `timestamp`  | HLC timestamp (hybrid logical clock)         |
| `before`     | Property snapshot before the change (if any) |
| `after`      | Property snapshot after the change (if any)  |
| `labels`     | Node labels: create/add post-image, remove/delete pre-image           |
| `lpg_graph` | Exact LPG graph path as a component array; absent/null for RDF |
| `triple_graph` | RDF named graph only; absent/null for RDF default or LPG |

LPG root is `lpg_graph: []`, distinct from `[""]`, `["default"]`,
`["a/b"]`, and `["a", "b"]`. RDF events never place their graph name in
`lpg_graph`; LPG events never place a coordinate in `triple_graph`.
Python/Node event dictionaries include null for absent coordinates. Rust's
`ChangeEvent::graph_path()` borrows the exact LPG path.

Entity-only history aggregates colliding LPG IDs across graph paths. Coordinates
also aggregate successive DROP/CREATE incarnations; they are not incarnation
handles. To select one exact model-specific coordinate in Rust:

```rust
use grafeo_common::types::{EpochId, GraphPath};
use grafeo_engine::cdc::{EntityHistoryQuery, HistoryGraph};

let mut query = EntityHistoryQuery::new(node_id);
query.graph = HistoryGraph::Lpg(GraphPath::from_components(&["a", "b"])?);
query.since_epoch = EpochId::new(42);
let page = db.session().history_after(&query, None, 128, 1024 * 1024)?;

// For a triple, select HistoryGraph::Rdf(None) for default or
// HistoryGraph::Rdf(Some("urn:graph".into())) for a named RDF graph.
```

`HistoryGraph::All` is the default and aggregates authorized coordinates for the
selected entity. Explicit graph requests require that exact read grant. The
entity index seeks directly to candidates; the row limit bounds inspected
candidates, including graph/epoch/permission-filtered rows. An exhausted index
can advance an empty page to the shared feed tail without scanning unrelated
events. Stop only when the cursor is unchanged. A cursor is not bound to the
selection: widening a query requires an earlier cursor or `None` to include
previously skipped events. `since_epoch` is inclusive.

Retention runs through database GC and removes complete oldest epochs under the
publication fence. The directory WAL persists and syncs a changed feed floor
before pruning; single-file checkpoints preserve it with the retained events.
Pruning feed events does not authorize deletion of native graph history or WAL
needed by a backup lease. Exact replicas preserve valid retained cursors;
logical forks have a new StoreId, and restoring an earlier cut rejects cursors
from its future. Final release performance and platform qualification remain open.

### Bounded feed pages

Both limits are required and positive. The row limit bounds inspected rows as
well as returned events; the byte limit covers native serialized event images.
Session pages apply read grants before copying or byte-counting hidden payloads.
A filtered page can be empty while its cursor advances. Stop only when the
returned cursor is unchanged, and retain it to resume later.

```rust
let mut cursor = None;
loop {
    let page = db.session().changes_after(cursor.as_ref(), 128, 1024 * 1024)?;
    if cursor == Some(page.next) { break; }
    cursor = Some(page.next);
    for event in page.events {
        // Apply any desired epoch filter, then consume this event.
    }
}
```

Cursors encode the shared store/feed, generation, sequence and epoch in 97
canonical bytes. They are resume coordinates, not authorization credentials.
Malformed, foreign and evicted cursors fail with `GRAFEO-S004`, `GRAFEO-S005`
and `GRAFEO-S006`. An individual visible event exceeding the byte budget fails
with `GRAFEO-S001`; retry with a sufficient limit. Returned pages own their data
and remain usable after database close. Reading a closed database fails.
Disabling future capture does not hide previously retained events.

### Python

```python
node_page = db.node_history_after(node_id, None, 128, 1024 * 1024, since_epoch=42)
edge_page = db.edge_history_after(edge_id, None, 128, 1024 * 1024)
# Resume either selection with its page["next"] bytes using the loop below.
cursor = None
while True:
    page = db.changes_after(cursor, max_events=128, max_bytes=1024 * 1024)
    if page["next"] == cursor:
        break
    cursor = page["next"]  # bytes; preserve exactly for the next request
    for event in page["events"]:
        if 10 <= event["epoch"] <= 50:
            consume(event)
```

### Node.js

```typescript
const nodePage = await db.nodeHistoryAfter(String(nodeId), null, 128, 1024 * 1024, "42");
const edgePage = await db.edgeHistoryAfter(String(edgeId), null, 128, 1024 * 1024);
// Resume either selection with its next Buffer using the loop below.
let cursor: Buffer | null = null;
for (;;) {
  const page = await db.changesAfter(cursor, 128, 1024 * 1024);
  if (cursor?.equals(page.next)) break;
  cursor = page.next;
  for (const event of page.events) {
    // IDs, epochs, HLC timestamps and graph incarnations are exact decimal strings.
    if (10n <= BigInt(event.epoch) && BigInt(event.epoch) <= 50n) consume(event);
  }
}
```

## Transaction semantics

CDC events are **buffered** during a transaction:

- On **commit**, buffered events are flushed to the CDC log with the commit
  epoch assigned.
- On **rollback**, buffered events are discarded.
- On **rollback to savepoint**, events after the savepoint are truncated.

This means the CDC log only contains events from successfully committed
transactions.

## Performance considerations

When CDC is disabled (the default), there is **zero overhead** on the mutation
path. No HLC timestamps are generated, no events are buffered, and no store
wrapping occurs.

When enabled, each mutation incurs:

- One HLC timestamp generation (`SystemTime::now()` syscall + atomic CAS)
- One event buffer push (mutex-protected `Vec`)
- Property snapshot allocations for before/after state

For bulk data loading or benchmarks, disable CDC or use `session_with_cdc(false)`
to avoid this overhead.

## Feature flag

CDC requires the `cdc` feature flag at compile time. The `lpg`, `ai` and `enterprise`
profiles include it. The `edge` (WASM) profile does not.

```toml
[dependencies]
grafeo = { version = "0.5", features = ["ai"] }  # includes cdc
```


### C and Go page ownership

C exposes `grafeo_changes_after`, `grafeo_node_history_after` and
`grafeo_edge_history_after`. Each successful call returns an owned
`GrafeoChangePage`; its event JSON and 97-byte cursor are borrowed until
`grafeo_free_change_page`. Copy the cursor before freeing the page to resume.
Go's `ChangesAfter`, `NodeHistoryAfter` and `EdgeHistoryAfter` copy bounded output
and release that native handle before returning, including on conversion errors.
Their page values remain usable after database close and require no early-stop
cleanup. See the C and Go binding READMEs for complete signatures and limits.

C and Node event JSON carries exact decimal coordinate strings; Go decodes them
as `uint64`, while Python returns exact integers. Creation payloads include node
labels, edge type and both endpoint IDs, captured at the event's commit. Readers
never reconstruct those fields from the graph's current state.

Dart and C# also consume the native bounded page API. Their returned pages own
managed events/cursor copies and remain usable after database close; no page
cleanup is needed. Dart uses `BigInt` coordinates and C# uses `ulong`. See the
[Dart caller](../../crates/bindings/dart/README.md#bounded-change-pages) and
[C# caller](../../crates/bindings/csharp/README.md#bounded-change-pages) for the
required bounds, canonical cursor and exact-number conventions.

WASM exposes bounded pages with the `cdc` feature (included in `full`). It uses
owned JS event objects and `Uint8Array` cursor copies, decimal-string coordinates,
and the active transaction Session. Existing LPG snapshot export/import preserves
feed identity; storing the snapshot durably remains the caller's responsibility.
See the [WASM caller](../../crates/bindings/wasm/README.md#bounded-change-pages-cdc-feature).

### Cross-language retained-cut control

The C,Go,Python,Node,Dart and C# tests share a native retained cut and a cursor
created before its floor advanced. Generate that witness with the existing C
control,keeping the same absolute environment variable for the language tests:

```bash
export GRAFEO_CDC_EVICTED_FIXTURE=/absolute/test-output/retained.grafeo
cargo test -p grafeo-c --all-features --lib cdc_entity_selectors_and_eviction_keep_native_error_categories
```

The output includes `retained.grafeo` and `retained.cursor`. Go,Python,Node,Dart and
C# copy the store into temporary test directories; all three bounded page readers
must return `GRAFEO-S006` for the stale cursor and still accept a fresh read.
Without the witness these interop tests explicitly skip. Release qualification
supplies and verifies the witness and requires zero skips. WASM separately tests
that snapshot import preserves the same retirement floor and error category.
