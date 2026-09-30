# grafeo

Go bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core and no required C dependencies.

## Requirements

- Go 1.22+
- CGO enabled (`CGO_ENABLED=1`)
- The `grafeo-c` shared library (`libgrafeo_c.so` / `libgrafeo_c.dylib` / `grafeo_c.dll`)

## Installation

```bash
go get github.com/GrafeoDB/grafeo/crates/bindings/go
```

## Quick Start

```go
package main

import (
	"context"
    "fmt"
    "log"

    grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

func main() {
    db, err := grafeo.OpenInMemory()
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create nodes
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

    // Query with GQL
    result, err := db.Execute("MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age")
    if err != nil {
        log.Fatal(err)
    }
    for _, row := range result.Rows {
        fmt.Printf("Name: %v, Age: %v\n", row["p.name"], row["p.age"])
    }
}
```

## Contexts, limits, and cancellation

Context-aware execution accepts parameters and an optional set of execution
options. Eager results default to a one-million-row and 64 MiB limit. Set a
pointer to a `uint64` to apply an explicit limit; a pointer to zero is a real
zero limit. `Language` selects the parser (`"gql"` is the default).

```go
ctx := context.Background()
maxRows, maxBytes := uint64(10), uint64(4096)
rows, err := db.ExecuteContext(ctx, "RETURN $n AS n", map[string]any{"n": 17},
    &grafeo.ExecutionOptions{MaxRows: &maxRows, MaxBytes: &maxBytes})
```

`NewQueryControl` starts its deadline immediately. A negative duration means
no deadline, zero expires immediately, and positive durations are rounded up
to milliseconds. A control is single-use: `Cancel` and `Close` are safe to
coordinate with execution, and cancellation handles remain valid while an
execution is finishing. Transaction execution accepts the same context,
parameters, and options through `tx.ExecuteContext`.

`Error.Code` preserves native structured codes such as `GRAFEO-Q003` (deadline)
and `GRAFEO-Q007` (cancellation). These errors also support `errors.Is` for
`context.DeadlineExceeded` and `context.Canceled`, respectively. Result-limit
errors preserve the database error family.

The default 64 MiB byte cap covers native and Go copies together. Native C
admission reserves one quarter before commit; Go conversion uses the remaining
three quarters.
Limits are checked before a mutation is published, so a denied statement is
rolled back while earlier statements in the same transaction remain intact.

## Streaming results

Use `ExecuteStreamContext` for lazy rows or chunks:

```go
stream, err := db.ExecuteStreamContext(ctx,
    "UNWIND range(1, 2500) AS i RETURN i", nil, nil)
if err != nil { log.Fatal(err) }
defer stream.Close()
for {
    row, err := stream.Next()
    if err != nil { log.Fatal(err) }
    if row == nil { break }
    fmt.Println(row)
}
```

Streaming has no default total-row cap. `MaxRows`, when supplied, caps total
rows; `MaxBytes` applies to each delivered row or chunk. `NextChunk(maxRows)`
caps one chunk at 1,024 rows. `Collect` applies the selected row limit (one million by default) and the
shared byte budget, and returns no partial result when collection fails.

`Close() error` is safe to call repeatedly; a terminal stream failure remains
the returned error. Closing an active stream cancels and joins its pull before
releasing native resources. Database and transaction operations may return
`ErrBusy` during active execution or close contention; a busy database close
leaves the handle usable for a later retry.

## Features

- GQL, Cypher, SPARQL, Gremlin and GraphQL query languages
- Full node/edge CRUD with property management
- ACID transactions with configurable isolation levels
- HNSW vector similarity search
- Property indexes for fast lookups
- Thread-safe for concurrent use

## Index owners

```go
label, dimensions := "Doc", uint(3)
owner, err := db.CreateIndex(grafeo.CreateIndexRequest{
    Kind: grafeo.VectorIndex, Label: &label, Property: "embedding",
    Dimensions: &dimensions,
})
// Check err, then use db.RebuildIndex(owner) or db.DropIndex(owner).
```

`Graph: nil` selects root; `Graph: []string{""}` selects one empty-named
component. Components are literal UTF-8 strings, including separators and NULs.
Property/BTree omit `Label`; Text/Vector require it. Pointer options preserve
absence versus explicit empty/zero. Drop returns `(false, nil)` only for absence;
missing rebuild and all engine failures return errors. Read/search APIs remain.

## Building the Shared Library

```bash
# From the Grafeo repository root:
cargo build --release -p grafeo-c --features full

# The library is at:
#   target/release/libgrafeo_c.so      (Linux)
#   target/release/libgrafeo_c.dylib   (macOS)
#   target/release/grafeo_c.dll        (Windows)
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Python Package](https://pypi.org/project/grafeo/)
- [npm Package](https://www.npmjs.com/package/@grafeo-db/js)


### Bounded change pages

The native library must include `cdc`, as the default `embedded` build does.
Enable new-session capture with `db.SetCDCEnabled(true)`. `ChangesAfter` reads
the feed; `NodeHistoryAfter(id, sinceEpoch, cursor, maxEvents, maxBytes)` and
`EdgeHistoryAfter` read indexed entity history. Epoch zero selects all retained
history. A nil cursor starts at the retained floor; non-nil cursors must be the
exact 97 returned bytes.

```go
var cursor []byte
for {
    page, err := db.ChangesAfter(cursor, 1, 4096)
    if err != nil { return err }
    if bytes.Equal(cursor, page.Next) { break }
    for _, event := range page.Events { fmt.Println(event.EntityID, event.Kind) }
    cursor = page.Next
}
```

Both bounds are required and positive. Bytes count native event encodings,
excluding the transport envelope. Empty filtered pages may advance their cursor;
only an unchanged cursor marks EOF. The returned events and cursor are owned Go
values; the native page is already freed, so early stop needs no extra close.

IDs, epochs, HLC timestamps, incarnations and endpoints are exact `uint64` values.
Property numbers use `json.Number`, and creation events include node labels and
edge type/endpoints. Structured `*grafeo.Error` codes preserve invalid/foreign/
evicted cursor and resource errors. Owned pages survive database close. Directory
stores support native reopen/resume; in-memory stores make no reopen promise.

For the cross-language eviction control, set `GRAFEO_CDC_EVICTED_FIXTURE` to an
absolute temporary `.grafeo` path. Run the C retention test
(`cargo test -p grafeo-c --all-features --lib cdc_entity_selectors_and_eviction`)
first, then run the Go tests with the same variable. The C test saves a retained
cut and its pre-retention cursor; Go must receive `GRAFEO-S006` from that exact
store. A Go-only run without this witness reports that one interop test skipped.
