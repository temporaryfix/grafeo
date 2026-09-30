# D1 — Report the durable commit outcome accurately

[Back to the starting guide](../REVIEW.md). **First decision:** is this two-case
failure contract a useful first contribution to [#498](https://github.com/GrafeoDB/grafeo/issues/498),
with any changes needed to failed-handle and retry behavior?

## What the caller observes

Both cases use a persistent **LPG** database in `Sync` mode and insert one node.
The shared test file also enables RDF features for its other cases.

| Injected failure | `commit()` / original handle | Nodes after reopen |
|---|---|---:|
| Before appending the commit marker | Error; prepared node stays invisible; handle is poisoned | 0 |
| Acknowledgment lost after the marker was synced | Error identifying an unknown outcome; handle is poisoned; no in-process publication | 1 |

The second row is why an error cannot always mean rollback or justify blindly
retrying the mutation. These are assertions in the review source, not a fresh
runtime result.

## Small reading set

1. Read the two LPG tests: [failure before the marker](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/c1_durability.rs#L162-L186)
   and [lost acknowledgment after the marker](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/c1_durability.rs#L220-L251).
2. Compare [upstream Session commit](https://github.com/GrafeoDB/grafeo/blob/22d39f24be01ecfed1d30093bc395e3f363df7ea/crates/grafeo-engine/src/session/mod.rs#L4173-L4243)
   with the review's [marker/error/publication boundary](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/src/session/mod.rs#L11430-L11493).
   The review separates preparation from installing the prepared state. An
   ambiguous WAL acknowledgment poisons the handle without writing a
   contradictory abort marker.
3. If needed, inspect [transaction preparation](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/src/transaction/manager.rs#L1452-L1482).
   The public `PreparedCommit` convenience wrapper is not the internal durable
   publication protocol.
4. To verify the failure positions, inspect the [pre-write injection](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-storage/src/wal/log.rs#L1351-L1394),
   [post-write acknowledgment injection](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-storage/src/wal/log.rs#L1632-L1642),
   [Sync-mode flush/sync](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-storage/src/wal/log.rs#L283-L323)
   and [commit-record sync requirement](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-storage/src/wal/record.rs#L1037-L1059).

## Integration boundary

Keep upstream's grouped `WalBuffer`, savepoint ordering, graph-qualified writes,
checkpoint controls and faster direct-write path. #498 already identifies a
route using that WAL group without changing its format. Start by adapting these
failure cases to those paths, then split validation/reservation from publication.

The full review method also prepares catalog, index, RDF and CDC state. Porting
that method wholesale would expand the scope. Mixed-model atomicity, new formats
and the larger transaction architecture require their own agreement. Measure
single-model direct-write cost before accepting the adapted implementation.

## Optional reproduction after source review

From the review root, using Rust 1.97.1:

```sh
cargo +1.97.1 test --locked -p grafeo-engine --no-default-features \
  --features lpg,gql,triple-store,sparql,wal,grafeo-file,testing-crash-injection \
  --test c1_durability lpg_commit_append_failure_is_unpublished_and_recovery_aborts_it \
  -- --exact --test-threads=1
```

Repeat with the filter
`lpg_lost_commit_ack_is_resolved_as_committed_by_recovery`. Each exact filter
should select one test. Zero selected tests is not a pass. Commands were checked
against test names and feature gates; they were not run on this export.
