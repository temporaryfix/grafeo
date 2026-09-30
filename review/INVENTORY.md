# Detailed review reference

[Start with one review brief](../REVIEW.md). This is optional background.
The 47 commits group one source snapshot by topic. The source reading links pin
`dc6b43b10adc758c16ab9bc01f3cde7283f83df6`; the final commit adds the review guide.

## Commit-by-commit reading order

Section numbers link directly to the source commits; the final row opens the
review guide. Counts are changed paths. These dependent sections are not
independently qualified cherry-picks. Some modules contain large inline test
suites; inspect both production behavior and those tests.

| Section | Topic | Changed paths |
|---|---|---:|
| [01](https://github.com/temporaryfix/grafeo/commit/5da1886a70664a8cd5d73303a224cde7ff5a31e7) | align workspace and coverage configuration | 5 |
| [02](https://github.com/temporaryfix/grafeo/commit/b44128efee3f5ecc693bf75ce63ee73a2eabcc5b) | define temporal identities and exact value contracts | 25 |
| [03](https://github.com/temporaryfix/grafeo/commit/d82387e099e5bd82cdb2dd767188094cd5b78e20) | add compressed vector string and adjacency formats | 15 |
| [04](https://github.com/temporaryfix/grafeo/commit/307c64b66efef2747285d09f7f5d210157537df4) | account execution buffers and chunk transport | 14 |
| [05](https://github.com/temporaryfix/grafeo/commit/f3d14901afb7835e94a4b26b993892bfb4f29b6f) | retain native entity and property history | 37 |
| [06](https://github.com/temporaryfix/grafeo/commit/c760c5af48e5d7eb72ae38aeae6d2e64802e8444) | prepare atomic data and index publication | 37 |
| [07](https://github.com/temporaryfix/grafeo/commit/6b3a7a8171f37ef3e9f19737429352c2e397af20) | encode retained histories and columnar adjacency | 34 |
| [08](https://github.com/temporaryfix/grafeo/commit/4864f61970be20abfe58ec794ffca4f84c65b96e) | preserve historical reads through layered mutation | 3 |
| [09](https://github.com/temporaryfix/grafeo/commit/40ab606e8391b140be8f649d321ac53c10eb95d1) | retain exact index ownership and historical search | 25 |
| [10](https://github.com/temporaryfix/grafeo/commit/781207114deb2cfb4cd1e34aec02bccf8d0495a8) | retain transactional search and index history | 12 |
| [11](https://github.com/temporaryfix/grafeo/commit/c3193e0b2fca24544bf06af68b51da39b400406b) | preserve typed terms histories and Ring state | 20 |
| [12](https://github.com/temporaryfix/grafeo/commit/ccb6e36a057dbf2c5fd1f50c0a73609f66eb8fb3) | preserve reachability edge identity and algorithm semantics | 15 |
| [13](https://github.com/temporaryfix/grafeo/commit/b172c4ad03fea17d8d20c9333001bbd0a2732888) | propagate exact values and execution failures | 38 |
| [14](https://github.com/temporaryfix/grafeo/commit/0abad0174c69d052fe028beb5c95a732a7c854b2) | authenticate records and retain resource ownership | 12 |
| [15](https://github.com/temporaryfix/grafeo/commit/5dcc1703b6c4356076094fa1b5928d260556bfa4) | stream exact accounted external sorting | 2 |
| [16](https://github.com/temporaryfix/grafeo/commit/ed953e7020c1e424241ab06b6706a9ef0e964ef2) | spill exact partition and distinct state | 12 |
| [17](https://github.com/temporaryfix/grafeo/commit/dc3e34e327d3027cc17183d689d5ec9e841022fe) | retain file ownership through publication and replacement | 12 |
| [18](https://github.com/temporaryfix/grafeo/commit/d0ccc7d248413a94abd8f8413496865aa5568ac6) | authenticate transaction and recovery boundaries | 89 |
| [19](https://github.com/temporaryfix/grafeo/commit/fe76492a18810c4f613d063930e5367f73480c9c) | bound asynchronous work and terminal ownership | 2 |
| [20](https://github.com/temporaryfix/grafeo/commit/966f66c87e7c54d66687d7628828c402b0eae80c) | prepare publication and track serializable conflicts | 10 |
| [21](https://github.com/temporaryfix/grafeo/commit/89d64443790b0620a79147d79ae071aa759422af) | coordinate catalog identity and native model ownership | 45 |
| [22](https://github.com/temporaryfix/grafeo/commit/5870988fd9f099e321f3f3dd16c479a598c327c3) | preserve exact catalogs indexes and retained state | 24 |
| [23](https://github.com/temporaryfix/grafeo/commit/a26bdb435068e71b669df91d114ee1fbd642e2f0) | connect atomic mutation and statement rollback | 22 |
| [24](https://github.com/temporaryfix/grafeo/commit/d589f1248d939c33fb82631a662e765935a4076e) | preserve parser and translator semantic contracts | 14 |
| [25](https://github.com/temporaryfix/grafeo/commit/6a413d1f800cf704d24601e9471305532507a7e1) | align indexed lookups paths and procedures | 43 |
| [26](https://github.com/temporaryfix/grafeo/commit/4224fc3c6fcf71ea8e6c5011afbed391e9675427) | preserve exact identities and expression semantics | 7 |
| [27](https://github.com/temporaryfix/grafeo/commit/c7a76d989b0c15f0f191ea567ee423aa1159caaa) | expose bounded streaming and cooperative cancellation | 34 |
| [28](https://github.com/temporaryfix/grafeo/commit/780cfa9495148483a47ed1a3e4b49354d5c0ef2c) | coordinate asynchronous framed I/O and cleanup | 3 |
| [29](https://github.com/temporaryfix/grafeo/commit/5de7677d5c4a709ef44ec5fb817872ec76bc32bc) | schedule owned asynchronous query sorts | 2 |
| [30](https://github.com/temporaryfix/grafeo/commit/e9db6af8086bd2e1f05baa3d242ca3843232fc92) | cover crash recovery hostile input and exact restore | 45 |
| [31](https://github.com/temporaryfix/grafeo/commit/818a510badb1f7d521146766586b0c1efe2cf803) | cover compaction retained reads and persistent copies | 38 |
| [32](https://github.com/temporaryfix/grafeo/commit/47b8bd5b0d7f0c9a87c4ce1cb6afce436df26ab6) | cover snapshot and serializable query isolation | 15 |
| [33](https://github.com/temporaryfix/grafeo/commit/52c517128395adf8b6ab9689b3a85ac4952be2a7) | cover transactions valid time semantics and recovery | 21 |
| [34](https://github.com/temporaryfix/grafeo/commit/8031f5cf4a7f1aac3794c9d1cd3906f1ffebef9c) | assert free endpoints parallel edges and lookup work | 10 |
| [35](https://github.com/temporaryfix/grafeo/commit/c17ee6f20f5c58fec666ce1e5e9e46ab741df1b6) | cover public query and model regressions | 47 |
| [36](https://github.com/temporaryfix/grafeo/commit/7e90218626eb645f3c51fff600efa6d79dc6b3ef) | expose exact backup chains and durable change pages | 15 |
| [37](https://github.com/temporaryfix/grafeo/commit/e31ba1d4e9b91f0043a8b8552a9e3d6960779b59) | share exact entity and execution contracts | 5 |
| [38](https://github.com/temporaryfix/grafeo/commit/9bd226d8ee9be5b39fe2a7e745a70f0d364486a4) | expose owned results execution control and durable pages | 17 |
| [39](https://github.com/temporaryfix/grafeo/commit/f53b107afaa89ce96a0601145611a648eeb9eaf2) | connect exact model and bounded execution callers | 37 |
| [40](https://github.com/temporaryfix/grafeo/commit/9e9a89c90e1d1239047014df22abd6deb28b901a) | connect exact model and bounded execution callers | 32 |
| [41](https://github.com/temporaryfix/grafeo/commit/6231c0e4602a92f8a1895e7db2853f0108410ffb) | expose model capabilities and controlled results | 20 |
| [42](https://github.com/temporaryfix/grafeo/commit/43d0ff1132f33c3059e44beff6aea62d463976c3) | connect Go Dart and CSharp owned callers | 57 |
| [43](https://github.com/temporaryfix/grafeo/commit/8395977cc28b4e7521431bd2e539209b1960901f) | expose canonical native capabilities and host examples | 11 |
| [44](https://github.com/temporaryfix/grafeo/commit/7d5f2baf47dcead457365c13ec813bc6b1579aeb) | align language fixtures with exact model contracts | 11 |
| [45](https://github.com/temporaryfix/grafeo/commit/9af532d2a2b665dd5edd25581756d1421c1d5b58) | describe retained history and current API boundaries | 57 |
| [46](https://github.com/temporaryfix/grafeo/commit/dc6b43b10adc758c16ab9bc01f3cde7283f83df6) | define native and language package checks | 43 |
| [47](../REVIEW.md) | focused maintainer and LLM review guide | 9 |

## Validation of this exported source

Completed locally for the review preparation:

- Rust 1.97.1 workspace formatting and syntax traversal: passed.
- Offline, locked Cargo workspace metadata: 14 packages, passed.
- Rust packaging script unit tests: 13 passed; these do not compile Rust packages.
- All 17 Cargo/config TOML files compare semantically with the implementation
  source after the neutral host-profile rename to `temporal-host`.
- Secret scanning and disclosure checks cover the selected source. The published
  source and outgoing history scans reported zero detected secrets.

The source has substantial existing regression and qualification work, but the
review export has not received a fresh full Rust/binding/platform test run.
Formatting and metadata checks are not a release qualification. Results from
other source revisions are not relabelled as results for this branch.

The extraction changes names, comments, synthetic fixture text and documentation.
It does not intentionally change graph algorithms, transaction protocols,
resource limits or on-disk formats. The host profile retains its original
capability set under a neutral name. Required licenses and contributor notices
are retained.

## Limits and decisions still open

- A full release/platform matrix is unfinished. Windows, some resource and
  performance controls, and package-level qualification still need completion.
- Cross-process spill quota, authenticated scavenging, owned asynchronous sorting
  and native aggregate resource work are present. Their remaining release and
  performance gates are not claimed complete; unfinished follow-ups are excluded.
- Historical visibility is not retention authority; a viewing epoch does not
  recover versions that have been collected.
- Unsupported query and conformance cases remain explicit. This is not a claim
  of full W3C compliance.
- API and format compatibility with upstream require individual agreement.
  Existing differences are review inputs, not authorization for further removals.
- Recent upstream release fixes must be reconciled with any adopted change.

The first decision is which capabilities and contracts Grafeo should adopt.
Review-sized merge candidates and their validation gates can then be prepared
against the agreed upstream branch.

The GitHub files from the public base are retained unchanged throughout the series. This branch
is an implementation review; it is not a package or release. The inherited upstream changelog remains historical;
this document describes the review proposal.
