---
title: grafeo-core
description: Core data structures crate.
tags:
  - api
  - rust
---

# grafeo-core

Core graph storage and execution engine.

## Graph Storage

```rust
use grafeo_core::graph::lpg::{LpgStore, NodeRecord, EdgeRecord};

let store = LpgStore::new();
let node_id = store.create_node(&["Person"]);
```

## Indexes

```rust
use grafeo_core::index::HashIndex;

let index: HashIndex<String, NodeId> = HashIndex::new();
index.insert("Alix".into(), node_id);
```

## Execution

```rust
use grafeo_core::execution::{DataChunk, ValueVector, SelectionVector};

let chunk = DataChunk::empty();
```

### Cooperative execution control

`QueryExecutionControl` is the single mutable lifecycle owner for one
execution. It derives a cancel-only `QueryCancellationHandle`, an
orchestration-only `QueryExecutionCheckpoint`, and check-only
`QueryCancellationToken`s for workers. Cancellation and deadline checks are
cooperative; they do not preempt a blocking call. After all workers and
required pre-commit cleanup have quiesced, the owner must call
`try_begin_commit` before durable commit, and `complete` only after successful
materialization and mandatory cleanup. Duplicate or contradictory lifecycle
transitions fail closed.

An explicit checkpoint opts `Executor` or `Pipeline` into checks around
operator calls and finalization. Legacy deadline-only execution keeps its
existing cadence. `QueryResourceContext::new_with_cancellation` and
`with_spill_root` install a worker token during construction so clones sharing
an execution ID and memory pool cannot diverge onto unrelated cancellation
states. Engine callers can supply execution control through
`Session::execute_with_options` and `Session::stream_with_options`.
Monotonic deadlines are unavailable on wasm32; explicit cancellation remains
portable there.

### Query resource contexts

`QueryResourceContext` is available in every feature profile. Each fresh
context owns one fair-share query memory pool backed by the database buffer
manager. Clones share the same pool and execution ID; separately constructed
contexts receive strictly increasing, nonzero process-local IDs.

```rust
use grafeo_common::memory::buffer::BufferManager;
use grafeo_core::execution::QueryResourceContext;

let manager = BufferManager::with_budget(64 * 1024 * 1024);
let resources = QueryResourceContext::new(manager)?;
let grant = resources.try_allocate(4096)?;

assert_eq!(grant.size(), 4096);
assert_eq!(resources.query_stats().allocated_bytes, 4096);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The ID is diagnostic only: it is not a transaction/commit ID, durable identity,
or cryptographic nonce, and failed setup may leave gaps. A returned grant must
remain alive for as long as its bytes are resident. `buffer_stats()` and
`query_stats()` are point-in-time observations, not reservations.

When `spill` is enabled, construct a context with
`QueryResourceContext::with_spill_root(buffer_manager, &root, token)`.
`ensure_spill_manager()` lazily admits one authenticated query leaf using that
context's execution ID and cancellation token; clones share the admission.
`SpillRoot::open` requires a `SpillRootAuthority`. Low-level callers can consume
its query lease with `SpillManager::from_query_lease`. Raw directory-based
manager constructors, arbitrary manager injection into a resource context, and
the former `OperatorMemoryContext` alias are unavailable.

Consumer callbacks use `register_consumer_scoped`; keep the returned RAII
registration alive for the whole operator lifetime. Equal diagnostic consumer
names do not share lifecycle identity. Resource-context sort paths charge row
capacities, stable-sort scratch, codec workspace, run catalogs and writer
backing. A denied input-row grant can spill a nonempty run and retry that row
once; continued denial remains a structured error. A resource context alone
does not establish a complete memory bound for every operator or route.

`SpillDiskQuota` enforces the query's logical framed-byte limit across staging,
poisoned and published files. `SpillDiskStats` reports reserved-live,
published-live and peak logical bytes. The separate root ledger retains
physical reservations across processes and restarts, including conservative
allocation and metadata allowances. Failed or uncertain cleanup keeps its debt;
capacity is released only after confirmed deletion and synchronization.

Engine configuration exposes `Config::with_max_query_spill_bytes` and
`Config::with_max_root_spill_bytes`; both require a configured `spill_path` and
the `spill` feature. The engine supplies store identity and root authentication,
uses authenticated encrypted records for encrypted databases, and runs bounded
authenticated scavenging when opening the root. Unknown, live or substituted
artifacts are preserved. Root admission currently supports Linux and macOS;
Windows admission remains open.

Session execution installs the shared resource context in owned and cached
physical plans, profiled execution and supported streaming routes. These
callers include RDF execution; spill configuration is no longer limited to
cache-ineligible GQL. Query PROFILE retains physical reservation and cleanup
debt observations after execution. This shared construction does not imply
complete bounded execution: blocking-sort/aggregate streaming admission,
remaining route/merge/failure coverage and peak-memory qualification are still
open. The async spill adapter uses the same framed operations and ownership,
but production async query/merge integration and its full acceptance matrix
remain unfinished.

## Note

This is an internal crate. The API may change between minor versions.
