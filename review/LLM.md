# A bounded review for an LLM

Copy the prompt below into a session with this repository available, or give it
the public [branch link](https://github.com/temporaryfix/grafeo/tree/review/engine-evolution).
Choose D1 (durability), P1 (paths) or H1 (history). D1 is the suggested first pass.
The manifest is a reading index; it contains no credentials or private source refs.

```text
Review candidate D1 from review/manifest.json. Read its brief first.

Purpose: help Grafeo's maintainer decide whether one focused contribution is
worth adapting to upstream. Assess the proposed contract and adoption boundary.
The supplied material is a proposal whose claims must be checked against code.

Baseline:
- Review code: dc6b43b10adc758c16ab9bc01f3cde7283f83df6
- Upstream main: 22d39f24be01ecfed1d30093bc395e3f363df7ea
- Upstream release/0.5.44: e7c48583f39306e862414ecfb721ca0adf5b2a97
Use these pins for the comparison; identify any newer ref separately.

Read only the selected brief and its initial source ranges. Use the manifest's
optional ranges if needed to resolve a specific uncertainty. Follow further
callees only when necessary, naming the reason. Do not begin with the entire
repository, all 47 commits, or the complete upstream issue inventory.

Check:
1. What exact behavior is asserted, and where is it implemented?
2. Which upstream behavior/fixes must survive adoption?
3. Is there a small test/contract contribution before an architectural port?
4. What incompatibility, cost or missing evidence could block that contribution?

Separate inspected assertions from observed runtime results. No new runtime
pass or performance result is supplied for this exported snapshot. Test commands
are reproduction candidates; do not build by default. If execution is requested,
report the exact source, features, selected test count and outcome. Zero tests
is not a pass. Existing filenames and prose are not evidence of passing behavior.

Work read-only. Do not change code, CI, issues, PRs, repository settings or refs.

Return at most 500 words:
- Recommendation: adopt the tests/contract, adapt the proposal, defer, or reject.
- Strongest evidence, with file/symbol or pinned line citations.
- Smallest plausible upstream slice and behavior to preserve.
- Unresolved risk or evidence needed; label inference clearly.
- One decision for Steven and the next action the contribution author should own.

Stop after this case. Do not expand the review unless asked. If a cited source
or test cannot be located, report that gap rather than filling it in from memory.
```

[Human starting guide](../REVIEW.md) · [Machine-readable map](manifest.json) ·
[Optional full inventory](INVENTORY.md)
