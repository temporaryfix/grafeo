# Grafeo implementation review

This branch shares implementation and regression cases for discussion with
Grafeo's maintainers. Three concrete examples provide a starting point:

- A failed WAL commit reports an error; recovery resolves whether it committed.
- Shortest paths retain real edge identity and parallel-path multiplicity.
- Historical property values and neighbors survive compaction and reopen.

**Start with [the five-minute review guide](REVIEW.md).** It proposes one initial
contribution and two alternatives. Each brief has a small source reading list,
exact expected assertions, an adoption boundary and a decision to make.

For an assisted review, copy the [LLM handoff](review/LLM.md). Its
[machine-readable map](review/manifest.json) pins source versions and reading
ranges so an agent can evaluate one case without surveying the entire branch.

The code is an unreleased development snapshot, version `0.0.1`. The examples
above describe source contracts; the full runtime/platform matrix has not been
rerun on this export. See [validation and limitations](review/INVENTORY.md#validation-of-this-exported-source).
Future contributions will target the agreed upstream release branch with
compatibility preserved. Mixed-model atomicity and wider architecture are
separate discussions.

For the broader picture, see the [upstream comparison](UPSTREAM_REVIEW.md) and
[commit inventory](review/INVENTORY.md). GitHub configuration from the public base
is retained. No GitLab configuration is introduced.

## Attribution

Grafeo was created by S.T. Grond and the Grafeo contributors. This review includes
subsequent development by Temporary Fix. Original notices and attribution remain
in [LICENSE](LICENSE), [NOTICE](NOTICE) and [CONTRIBUTORS.md](CONTRIBUTORS.md).
