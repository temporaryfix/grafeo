---
title: grafeo-engine
description: Database engine crate.
tags:
  - api
  - rust
---

# grafeo-engine

Main database facade and coordination.

## GrafeoDB

```rust
use grafeo_engine::{GrafeoDB, Config};

// In-memory
let db = GrafeoDB::new_in_memory();

// Persistent
let db = GrafeoDB::open("path/to/db")?;

// With config
let config = Config::in_memory()
    .with_memory_limit(4 * 1024 * 1024 * 1024)
    .with_threads(8);
let db = GrafeoDB::with_config(config)?;
```

## Session

```rust
let mut session = db.session();

session.execute("INSERT (:Person {name: 'Alix'})")?;

let result = session.execute("MATCH (p:Person) RETURN p.name")?;
for row in result.rows() {
    println!("{:?}", row);
}
```

## Result limits and ownership

Eager queries default to 1,000,000 rows and 64 MiB of retained row storage.
Set database defaults with `Config::with_result_limits`, or override them for
one execution:

```rust
use grafeo_engine::{ExecutionOptions, ResultLimits};

let result = db.execute_with_options(
    "MATCH (p:Person) RETURN p.name",
    Default::default(),
    ExecutionOptions {
        result_limits: Some(ResultLimits { max_rows: 100, max_bytes: 1024 * 1024 }),
        ..Default::default()
    },
)?;
let rows = result.into_rows()?;
```

Limits admit container capacity and nested values before retaining them. Output
schema and status metadata also consume the database's memory grant. Zero limits
permit an empty row set; metadata still needs a grant. Larger explicit limits
remain constrained by available grants. Exhaustion returns `StorageFull`, with
no partial eager result. Output denial rolls back the current mutation statement
before commit, preserving earlier work in an explicit transaction.

Owned rows and dense columns retain their reservation, including individual
items moved from their iterators. Borrowed `rows()` also supports dense results;
its materialization space is reserved during collection.

Native streams support fallible `next_chunk` and explicit `close`. Each returned
`StreamChunk` owns its output reservation and exposes fallible borrowed `rows()?`,
`columns()` and `column_types()`. Drop chunks as they are consumed to release
their grants; retained chunks remain charged after stream or database close.
Keeping too many chunks can exhaust the shared query grant. The row iterator
retains its buffered chunk reservation, but returned `Vec<Value>` rows are
independently owned and do not retain that grant. To collect,
call `stream.collect(ResultLimits { max_rows: 100, max_bytes: 1024 * 1024 })?`.
A collection limit cannot loosen stricter limits supplied in stream options.
Mutation streaming remains explicitly unsupported; execute mutations eagerly
so returned rows are admitted before commit. Native result accounting does not
yet establish end-to-end limits for every operator or language conversion; those
qualification gates remain open.

## Transactions

```rust
let mut session = db.session();
session.begin_transaction()?;
session.execute("...")?;
session.commit()?;
// or
session.rollback()?;
```

### Stored procedures

Create and call catalog procedures through `Session`. Procedure definitions
participate in explicit transactions, and calls inherit the caller's private
catalog view, authorization, transaction or auto-commit boundary, statement
rollback, constraints and configured WAL/CDC machinery.

```rust
use grafeo_common::types::Value;

let session = db.session();
session.execute(
    "CREATE PROCEDURE plant(name STRING) RETURNS (name STRING) AS { \
     INSERT (n:Plant {name: $name}) RETURN n.name AS body_local }"
)?;
let result = session.execute_with_params(
    "CALL plant($value) YIELD name",
    [("value".to_string(), Value::String("fern".into()))].into(),
)?;
assert_eq!(result.columns, ["name"]);
```

`RETURNS` defines the public output names and types; body columns map to them
positionally. `YIELD` selects declared names, optionally with aliases. Invalid
output selections or return values fail within the statement boundary, before
commit. Read-only sessions can call read-only procedures; mutating calls require
write authority. See [transactions](../../user-guide/transactions.md).

## Bounded native CDC

With the `cdc` feature, `Config::with_cdc()` enables transaction event capture.
`GrafeoDB::changes_after` and `Session::changes_after` take an optional durable
cursor plus positive event and byte limits, returning an owned `ChangePage`.
`history_after` adds an `EntityHistoryQuery` with entity, graph and epoch filters.
Session reads enforce grants before copying payloads; a filtered empty page may
still advance its cursor. Stop when the cursor is unchanged.

Current WAL/container/snapshot formats preserve the retained feed and cursor
identity. In-memory state has process lifetime unless snapshotted. GC may expire
old cursors; handle structured `CursorEvicted`, `CursorForeign` and
`CursorInvalid` errors rather than silently starting over. These APIs expose a
retained window, not permanent history or external delivery acknowledgement.
See [the CDC guide](../../user-guide/cdc.md) for payload and ownership rules.
