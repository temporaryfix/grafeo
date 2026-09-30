# Start with one contribution

Steven, the useful first step is to select one contract worth bringing upstream.
The branch contains a substantial implementation, but each brief below stands
on its own as a review task. Choosing one leaves the rest for later discussion.

## Suggested first review: WAL acknowledgment

Your [#498](https://github.com/GrafeoDB/grafeo/issues/498) already specifies the
ordering we want: validate, write and acknowledge the WAL group, then publish.
The [durability brief](review/DURABILITY.md) shows two small LPG cases: failure
before the marker, and lost acknowledgment after the marker is durable. Their
recovery outcomes differ. It links the tests and the relevant commit boundary.

**Proposed first contribution:** adapt those cases to your grouped WAL and direct
write paths, then prepare the smallest implementation change needed for #498.
The broader mixed-model implementation is useful reference material; adopting it
is a separate decision. The contribution author will prepare and qualify the
adapted patch against the agreed release base.

For this first pass, the useful response is whether that test/contract slice is
worth taking, and whether the failed-handle/retry behavior needs changing. Reading
the brief and its two test functions is enough to begin that discussion.

## Two alternatives

| Review | Concrete example | First adoption boundary |
|---|---|---|
| [Path identity and multiplicity](review/PATHS.md) | Five free targets become nine distinct shortest paths after one parallel edge is added. | Regression cases and output reconstruction for #318; preserve #516's fixes. |
| [History after compaction and reopen](review/HISTORY.md) | Current value is 51.6, retained value is 51.5; a deleted edge remains visible only at the earlier epoch. | A persistence contract for retained history; agree retention/API behavior separately. |

## If using an LLM

[Copy this prompt](review/LLM.md). It defaults to the durability brief and asks for
a bounded assessment with source citations, preserved upstream behavior, blockers
and one decision. It includes an explicit stopping point and does not request a
repository-wide audit or a build by default.

## Scope and evidence

These are source-backed review candidates. Their test assertions and commands
are identified; fresh runtime results are not claimed. The initial export passed
formatting, Cargo metadata and 13 packaging-script tests. The full regression and
performance matrix remains to be qualified for any adapted contribution.

The [upstream comparison](UPSTREAM_REVIEW.md) checks main `22d39f24` and
release/0.5.44 `e7c48583`; the source snapshot is `dc6b43b1`. The 47 topic commits
are dependent review sections, not separately qualified cherry-picks. The
[detailed inventory](review/INVENTORY.md) is available when useful.
