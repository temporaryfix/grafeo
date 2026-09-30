# H1 — Keep retained history through compaction and reopen

[Back to the starting guide](../REVIEW.md). **First decision:** is this retention
and persistence case useful as an upstream regression contract while the storage
and epoch APIs evolve? It complements [#389](https://github.com/GrafeoDB/grafeo/issues/389)
and [#449](https://github.com/GrafeoDB/grafeo/issues/449).

## Concrete result

Commit a property value of `51.5` at epoch `e1`, then update it to `51.6`.
Create two incoming edges, remember that epoch, then delete one edge. Compact,
close and reopen the database without another compaction.

The source assertions require:

- Current property value: `51.6`; the same property at `e1`: `51.5`.
- At the earlier edge epoch: both neighbors.
- At the deletion epoch: only the surviving neighbor.

This is an LPG caller using native APIs. It can be reviewed independently of
cross-model transactions or a query-language design.

## Small reading set

1. [Compact/close/reopen case](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/temporal_host.rs#L180-L271):
   `persist_as_of_after_compact_save_open_without_recompact`.
2. [Public historical property getter](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/src/database/crud.rs#L312-L332)
   dispatches to the layered store when present.
3. [Overlay/base historical lookup](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-core/src/graph/compact/layered.rs#L3699-L3733)
   checks whether the overlay actually covers the requested epoch before falling
   back to the retained base version.
4. Optional broader case: [recompact/exact-copy timeline](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/temporal_persistence_fidelity.rs#L1223-L1246)
   includes closed entity lifetimes, labels and properties.

## Integration boundary

Keep the contract portable to upstream's planned compact-core store. Its physical
representation need not match this branch's base/overlay layout. First agree
what retained epochs callers may request, how retention is configured, and how
collected history is reported. A saved epoch alone does not prevent collection.

This case does not establish snapshot isolation or serializability: those need
the concurrent reader/writer and index tests associated with
[#412](https://github.com/GrafeoDB/grafeo/issues/412). It also does not prescribe
unbounded history, a retention default or incompatible format changes.

## Optional reproduction after source review

```sh
cargo +1.97.1 test --locked -p grafeo-engine --no-default-features \
  --features temporal-host --test temporal_host \
  persist_as_of_after_compact_save_open_without_recompact \
  -- --exact --test-threads=1
```

The profile supplies LPG, compact storage, statements and WAL/file support.
The exact filter should select one test. The command was checked against source
feature gates but has not been run on this export.
