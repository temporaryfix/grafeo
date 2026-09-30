---
title: grafeo-common
description: Foundation types crate.
tags:
  - api
  - rust
---

# grafeo-common

Foundation types, memory allocators and utilities.

## Types

```rust
use grafeo_common::types::{NodeId, EdgeId, Value, LogicalType};
```

### NodeId / EdgeId

```rust
let node_id = NodeId(42);
let edge_id = EdgeId(100);
```

### Value

```rust
let v = Value::Int64(42);
let v = Value::String("hello".into());
let v = Value::List(vec![Value::Int64(1), Value::Int64(2)].into());
```

### LogicalType

```rust
let t = LogicalType::Int64;
let t = LogicalType::String;
let t = LogicalType::List(Box::new(LogicalType::Int64));
```

## Memory

```rust
use grafeo_common::memory::{Arena, ObjectPool};
use grafeo_common::types::EpochId;

let arena = Arena::new(EpochId(0));
let data = arena.alloc_value(MyStruct::new());
```

### Memory-grant migration in 0.5.43

`grafeo-common` is an internal, unstable crate. Its 0.5.43 accounting changes
are source-impacting for applications that depend on it directly; the stable
`grafeo` and `grafeo-engine` facade APIs are unchanged.

| Before 0.5.43 | 0.5.43 replacement |
|---|---|
| Implement or call the public `GrantReleaser` raw accounting methods | Allocate only through `BufferManager::try_allocate` or `QueryMemoryPool::try_allocate`; downstream custom accounting implementations are no longer a public extension point. |
| `MemoryGrant::consume() -> usize` | `MemoryGrant::detach() -> DetachedMemoryGrant`, then keep the RAII token, `reattach()`, or explicitly `release()`. |
| `CompositeGrant::consume_all() -> usize` | `try_detach_all()` returns one opaque `DetachedCompositeGrant` that can be reattached or released without losing child accounting authority. |
| `MemoryGrant::merge(other)` for transfer | Use `try_merge(other)` and handle rejection. Grants must have the same region and exact backing account; the compatibility `merge` method panics on rejection. |
| `MemoryGrant::resize(new_size) -> bool` when the denial reason matters | Use `try_resize(new_size) -> Result<(), MemoryGrantError>` for global/query limit, overflow, or poisoned-account detail. The boolean method remains available. |

Detached `release()` methods are explicit lifetime control equivalent to
dropping the token, not a fallible durability acknowledgement. To observe a
structured single-grant reconciliation result, reattach the token and call
`try_resize(0)` before dropping it.

## Utilities

```rust
use grafeo_common::utils::{FxHashMap, FxHashSet};

let map: FxHashMap<String, i64> = FxHashMap::default();
```
