# Feature Profiles

## Motivation

Grafeo aims to be a universal graph database: property graphs, RDF, analytics, AI memory, browser, production server. But no single user needs all of that. Feature profiles let every user get exactly what they need, with zero overhead from what they don't.

There are two layers:

- **Layer 1: Profiles**, named groups consistent across the entire ecosystem. This is what most users interact with.
- **Layer 2: Atoms**, individual feature flags for power users who want precise control. Profiles are composed from these.

## Current Profiles

The current system uses persona names:

| Profile | Contents | Use case |
| --- | --- | --- |
| `lpg` | LPG and LPG query languages with storage | Graph applications |
| `rdf` | RDF, SPARQL, GraphQL, ring index, storage, regex, SHACL | Knowledge applications |
| `analytics` | Algorithms, search, and bulk import | Data science |
| `ai` | Vector/text/hybrid search and CDC | AI and agent applications |
| `edge` | LPG, GQL, compact store, regex-lite | WASM and resource-constrained applications |
| `enterprise` | Metrics, tracing, and async storage | Platform operations |

**Defaults**: the grafeo facade uses its explicit embedded capability expansion; native bindings retain their `embedded` default; WASM uses `edge`; CLI uses `gql` + storage.

---

## Persona profiles

The facade's deployment-target aliases have been removed. WASM uses `edge`; its `full` convenience group remains available. Native binding aliases remain until their own compatibility migration. `temporal-host` is a **compile slice** for an embedded temporal host — the engine is still LPG+RDF; add `triple-store`/`sparql` for RDF engine builds.

| Profile | Persona | What it enables |
| --- | --- | --- |
| **LPG** | Graph App Developer | GQL, Cypher, Gremlin, SQL/PGQ, storage |
| **RDF** | Knowledge Engineer | SPARQL, GraphQL, RDF store, ring index, SHACL, storage, regex |
| **Analytics** | Data Scientist | Algorithms, vector/text/hybrid search, parquet + jsonl import |
| **AI** | AI Memory / Agent Developer | Native temporal/as-of history, CDC, vector/text/hybrid search |
| **Edge** | Frontend / Edge Developer | LPG, GQL, compact store, regex-lite |
| **Temporal-host** | Embedded temporal host | LPG + compact-store + statement-table + WAL + text-index. Add `sparql`/`gql` when the host wants languages. |
| **Native** | Parser-free dual model | LPG CRUD + RDF quads + WAL + `.grafeo`. No GQL/SPARQL/Cypher/Gremlin/GraphQL/SQL-PGQ. |
| **Enterprise** | Platform Operator | Facade atoms: metrics, tracing, async storage. Server product: auth, TLS, sync, replication, push changefeeds, and transports. |

### Composition Rules

- **Model profiles** (LPG, RDF): pick one or both. These are the foundation.
- **Capability profiles** (Analytics, AI, Enterprise): stack on top of a model profile.
- **Constrained profile** (Edge, Temporal-host): minimal by default, composable with constraints. Users can add atoms like `algos` if they accept the size increase.

Examples:

```toml
# AI memory developer
grafeo = { features = ["lpg", "ai"] }

# Semantic data scientist
grafeo = { features = ["rdf", "analytics"] }

# Full production stack
# grafeo-server = { features = ["lpg", "rdf", "analytics", "ai", "languages", "parallel", "arrow-export", "enterprise"] }

# Embedded temporal host (add sparql / gql for an RDF or LPG query session)
grafeo = { default-features = false, features = ["temporal-host"] }

# Parser-free LPG + RDF (insert_rdf_quads / create_node, no query languages)
grafeo = { default-features = false, features = ["native"] }

# Browser app
grafeo = { features = ["edge"] }

# Power user: just Cypher + vector search
grafeo = { features = ["cypher", "vector-index", "wal"] }
```

## Profile Definitions

### LPG

```toml
lpg = ["gql", "cypher", "gremlin", "sql-pgq", "storage", "regex"]
```

All Labeled Property Graph query languages plus persistence. The default choice for application developers working with nodes, edges, labels, and properties.

### RDF

```toml
rdf = ["triple-store", "sparql", "graphql", "ring-index", "storage", "regex", "shacl"]
```

The comprehensive knowledge-engineering profile: RDF storage, SPARQL and
GraphQL query surfaces, compact ring indexing, SHACL validation, persistence,
and full regex support. It does not enable the LPG model or LPG query
languages. For a smaller parser-free or SPARQL-only host, compose the
`triple-store`, `sparql`, `wal`, and `grafeo-file` atoms directly.

> **Note:** `owl-schema` and `rdfs-schema` currently only exist in grafeo-server. Promoting them to the engine level for this profile is an open question.

### Analytics

```toml
analytics = ["algos", "vector-index", "text-index", "hybrid-search", "jsonl-import", "parquet-import"]
```

27 graph algorithms (PageRank, Louvain, SSSP, Dijkstra, BFS/DFS, centrality, community detection, MST, flow, isomorphism, structural analysis, clustering), search indexes, and data import. Combine with LPG or RDF depending on the dataset.

### AI

```toml
ai = ["vector-index", "text-index", "hybrid-search", "cdc"]
```

Structured memory for LLMs, agents, and RAG pipelines. Native transaction-time
and as-of history is part of the store rather than a removable Cargo feature;
CDC enables change feeds, and the search indexes support vector/text retrieval.

> **Note:** `embed` (ONNX embedding generation, ~17 MB) is deliberately excluded from this profile. Most AI memory use cases (grafeo-memory, MCP, LangChain) bring embeddings via API calls. Opt in explicitly with `features = ["ai", "embed"]` if you need in-process embedding.

### Temporal-host

```toml
temporal-host = ["lpg", "compact-store", "statement-table", "wal", "text-index"]
```

Embedded host slice: CompactStore as-of hops (`fill_neighbors_at_epoch` / `fill_neighbors_of_types_at_epoch`) and opaque statement rows. Default compile set omits `gql`/`gremlin`/`graphql`/`cdc`/`algos` so a small host binary stays small. RDF and SPARQL remain first-class features in the same engine. Add language atoms when the host wants them. Host test: `temporal_host`.

### Native

```toml
native = ["lpg", "triple-store", "wal", "grafeo-file"]
```

Parser-free dual native model: LPG `create_node` / `create_edge` and RDF `insert_rdf_quads` / `contains_rdf_quad` on `GraphModel::Both`, with WAL and `.grafeo`. Does not compile GQL, SPARQL, Cypher, Gremlin, GraphQL, or SQL/PGQ. Bindings (`grafeo-c`, Python, Node, WASM) take `grafeo-engine` with `default-features = false` so `--features native` does not smuggle engine `gql`. Engine test: `native_no_parsers`. Binding check: `cargo check -p grafeo-c --no-default-features --features native`.

### Edge

```toml
edge = ["lpg", "gql", "compact-store", "regex-lite"]
```

Compact facade profile for browser, mobile, and resource-constrained
environments. It includes the columnar compact store and lightweight regex;
the WASM build additionally enforces its gzip budget.

### Enterprise

```toml
enterprise = ["metrics", "tracing", "async-storage"]
```

Embedded operational capabilities: metrics, tracing, and asynchronous storage.
Authentication, TLS, replication, push changefeeds, and server transports are
not enabled by this facade profile; they remain server-level product
capabilities rather than being retired by this feature-map correction.

Cargo packaging is not a capability-retirement mechanism. The facade feature
only selects code present in this workspace; the server product's stronger
operational contract remains a target contract. Removing any of those
capabilities, or narrowing an existing profile rather than moving a capability
to an explicitly composable atom, requires a separate compatibility decision.

## Migration from Current Profiles

| Old Profile | New Equivalent | Notes |
| --- | --- | --- |
| Facade default | `lpg-model` + `gql` + `ai` + `algos` + `parallel` + `regex` + JSONL/Arrow I/O | Grafeo facade capability expansion |
| Native binding default | `embedded` | Python/Node/C binding default |
| `edge` | `edge` | Current default for WASM |
| Production stack | `lpg` + `rdf` + `analytics` + `ai` + `languages` + `parallel` + `arrow-export` + `enterprise` | Full facade capability expansion; heavyweight `embed` remains opt-in |

The facade default uses its explicit embedded capability expansion. Native binding defaults retain `embedded`; persona names do not silently alter that default.

## Ecosystem Matrix

The profile names are consistent across every project. The table below shows which profiles are available in each project, either as configurable feature flags or as the project's inherent profile alignment.

### Core Engine

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo** (facade) | flag | flag | flag | flag | flag | flag | Enterprise atom means metrics + tracing + async storage |
| **grafeo-server** | flag | flag | flag | flag | n/a | flag | Product profile adds auth, TLS, sync, replication, push changefeeds, and transports |
| **grafeo-cli** | flag | flag | flag | flag | n/a | n/a | Interactive REPL and CLI tooling |

### Language Bindings

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Python** (grafeo-py) | flag | flag | flag | flag | n/a | n/a | Default: `embedded` |
| **Node.js** (grafeo-node) | flag | flag | flag | flag | n/a | n/a | Default: `embedded` |
| **WASM** (grafeo-wasm) | flag | flag | flag | n/a | flag (default) | n/a | Edge is default |
| **C** (grafeo-c) | flag | flag | flag | flag | n/a | n/a | Default: `embedded`; bridge for C#, Dart, Go |
| **C#** | via C | via C | via C | via C | n/a | n/a | Feature selection at C build time |
| **Dart** | via C | via C | via C | via C | n/a | n/a | Feature selection at C build time |
| **Go** | via C | via C | via C | via C | n/a | n/a | Feature selection at C build time |

### AI / Agent Ecosystem

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo-memory** | inherent | - | - | inherent | - | - | AI memory layer |
| **grafeo-langchain** | inherent | - | - | inherent | - | - | LangChain integration |
| **grafeo-llamaindex** | inherent | - | - | inherent | - | - | LlamaIndex integration |
| **grafeo-mcp** | inherent | - | - | inherent | - | - | MCP server for AI agents |

### Web / Visualization

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo-web** | - | - | - | - | inherent | - | WASM in browser |
| **playground** | - | - | - | - | inherent | - | Interactive graph playground |
| **anywidget-graph** | inherent | - | - | - | - | - | Notebook graph visualization |
| **anywidget-vector** | - | - | - | inherent | - | - | Notebook vector visualization |

### Protocol Libraries

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **boltr** | - | - | - | - | - | inherent | Bolt v5 protocol |
| **gwp** | - | - | - | - | - | inherent | GQL Wire Protocol (gRPC) |

### Accelerators and Tooling

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo-cuda** | - | - | inherent | - | - | - | GPU-accelerated algorithms |
| **graph-bench** | all | all | all | all | - | - | Benchmark suite |

### Legend

- **flag**: Profile is available as a configurable feature flag. User opts in.
- **inherent**: The project is inherently aligned with this profile. No flag needed.
- **via C**: Feature selection happens at C binding compile time, propagates to higher-level bindings.
- **n/a**: Profile does not apply to this project.
- **-**: Not applicable or not supported.

## Atom Reference

The complete list of individual feature flags (Layer 2) that profiles are composed from. Status indicates whether the atom exists in the codebase today.

### Query Languages

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `gql` | LPG, Edge | ISO/IEC GQL standard | Implemented |
| `cypher` | LPG | openCypher 9.0 | Implemented |
| `sparql` | RDF | SPARQL 1.1 target; unsupported cases fail closed until the official matrix is green | Qualification in progress |
| `gremlin` | LPG | Apache TinkerPop | Implemented |
| `graphql` | RDF / `languages` | GraphQL query parser | Implemented |
| `sql-pgq` | LPG | SQL:2023 GRAPH_TABLE | Implemented |

### Storage

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `storage` | LPG, RDF | Umbrella: WAL + grafeo-file + spill + mmap | Implemented |
| `wal` | (storage) | Write-ahead log persistence | Implemented |
| `grafeo-file` | (storage) | Single-file .grafeo format | Implemented |
| `spill` | (storage) | Out-of-core disk spilling | Implemented |
| `mmap` | (storage) | Memory-mapped file storage | Implemented |
| `async-storage` | (standalone) | Async WAL backend (tokio) | Implemented |
| `compact-store` | Temporal-host, Edge | Layered columnar store | Implemented |

### Graph Model

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `triple-store` | RDF | Native RDF triple/quad store | Implemented |
| `ring-index` | RDF | Space-efficient RDF index (also available as an atom) | Implemented |
| `succinct-indexes` | (pulled in by ring-index) | Rank/select bitvectors, Elias-Fano, wavelet trees | Implemented |
| `owl-schema` | RDF (server only) | OWL schema loading | Server only |
| `rdfs-schema` | RDF (server only) | RDFS schema support | Server only |

### Search and AI

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `vector-index` | Analytics, AI | HNSW approximate nearest neighbor | Implemented |
| `text-index` | Analytics, AI | BM25 inverted index | Implemented |
| `hybrid-search` | Analytics, AI | Combined vector + text search | Implemented |
| `embed` | (standalone) | ONNX embedding generation (~17 MB overhead) | Implemented |
| `algos` | Analytics | 27 graph algorithms | Implemented |

### Temporal and Change Tracking

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `temporal` | AI | Append-only versioned properties | Implemented |
| `cdc` | AI | Change data capture with history API | Implemented |

### Import

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `jsonl-import` | Analytics | JSON Lines file import | Implemented |
| `parquet-import` | Analytics | Apache Parquet import | Implemented |

### Execution

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `parallel` | (standalone) | Morsel-driven parallelism (rayon) | Implemented |
| `tiered-storage` | (standalone) | Hot/cold version storage with epochs | Implemented |

> **Note:** Block-STM parallel transaction execution is compiled unconditionally (see `grafeo-engine/src/transaction/parallel.rs`). It is not gated behind a feature flag.

### Operations (grafeo-server only)

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `auth` | Enterprise | Authentication provider | Server only |
| `tls` | Enterprise | TLS/HTTPS encryption | Server only |
| `metrics` | Enterprise | Lock-free query/transaction metrics | Implemented |
| `tracing` | Enterprise | Distributed tracing spans | Implemented |
| `sync` | Enterprise | Pull-based changefeed for offline-first | Server only |
| `push-changefeed` | Enterprise | Push-based SSE/WebSocket changefeed | Server only |
| `replication` | Enterprise | Primary-replica replication | Server only |

### Transports (grafeo-server only)

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `http` | Enterprise | HTTP/REST + OpenAPI + WebSocket | Server only |
| `gwp` | Enterprise | GQL Wire Protocol (gRPC) | Server only |
| `bolt` | Enterprise | Bolt v5 (Neo4j driver compat) | Server only |
| `studio` | Enterprise | Embedded web UI | Server only |

### Regex

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `regex` | LPG, RDF | Full regex engine | Implemented |
| `regex-lite` | Edge | Lightweight regex for WASM | Implemented |
