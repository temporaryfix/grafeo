# P1 — Preserve path identity and parallel-path multiplicity

[Back to the starting guide](../REVIEW.md). **First decision:** should we adapt
these cases and path reconstruction to upstream's current shortest-path
operator for [#318](https://github.com/GrafeoDB/grafeo/issues/318)? A shared
traversal replacement would be a separate review.

## Concrete result

The shared fixture contains a diamond, cycle, self-loop, disconnected component
and parallel edges. The query leaves its destination unbound:

```cypher
MATCH p = shortestPath((a:Node {id: 's'})-[:REL*1..2]->(b))
RETURN b.id, [n IN nodes(p) | n.id], edges(p)
```

The assertion expects five targets: `a,b,d,g,h`. Add a parallel `s → a` edge,
switch to `allShortestPaths`, and the expected result is nine paths:
`a×2, b×1, d×3, g×1, h×2`. Their edge-ID sequences must be distinct and resolve
to edges connecting the returned nodes. An unreachable target contributes no row.

## Small reading set

1. [Shared fixture and independent reference traversal](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/support/mod.rs#L31-L159).
2. [Free-endpoint and parallel-edge test](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/path_semantics.rs#L731-L820):
   `cypher_shortest_paths_cover_free_targets_and_parallel_edges`.
3. [Pinned identity control](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/path_semantics.rs#L824-L901):
   `cypher_shortest_path_exposes_raw_edge_values`. A misleading `_id:999`
   property must not replace the real edge identity.
4. If implementation detail is needed: [planner output columns](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/src/query/planner/lpg/expand.rs#L206-L263)
   and [path materialization](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-core/src/execution/operators/variable_length_expand.rs#L646-L704).

## Integration boundary

Preserve [#516](https://github.com/GrafeoDB/grafeo/pull/516): unreachable-pair
suppression, hop bounds, OPTIONAL null rows and later input rows. Also preserve
[correlation](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-engine/tests/path_semantics.rs#L306-L323) and
[snapshot/read-tracking](https://github.com/temporaryfix/grafeo/blob/dc6b43b10adc758c16ab9bc01f3cde7283f83df6/crates/grafeo-core/src/execution/operators/variable_length_expand.rs#L2978-L3050) controls.

This snapshot uses a shared expand operator and raw integer edge IDs in these
results. #318 asks for richer graph-element access too. Agree the public result
shape before porting; the cases do not establish that #318 is fully solved.

[#463](https://github.com/GrafeoDB/grafeo/issues/463)'s endpoint-only DISTINCT
optimization is separate. Deduplicating reachable nodes must not collapse the
nine observed paths above. No speedup is claimed by this brief.

## Optional reproduction after source review

```sh
cargo +1.97.1 test --locked -p grafeo-engine --no-default-features \
  --features lpg,cypher --test path_semantics cypher_shortest_
```

The filter selects the two named public tests. `cypher` enables `gql`, used by
the fixture setup. This proposed command has not been run on this export;
zero selected tests is not a pass.
