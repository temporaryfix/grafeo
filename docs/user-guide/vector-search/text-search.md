---
title: Text Search (BM25)
description: Full-text keyword search with BM25 scoring and inverted indexes.
tags:
  - text-search
  - bm25
  - full-text
---

# Text Search (BM25)

Grafeo includes a built-in BM25 full-text search engine with Unicode tokenization and stop word removal. Text indexes let you find nodes by keyword relevance without embeddings.


Text owners, tokenizer configuration and retained index history survive current-format
WAL recovery, checkpoints and whole-database copies. Retired owner IDs are not reused.

## Prerequisites

Text search requires the `text-index` feature flag, which is included in the `ai` and `analytics` profiles.

## Creating a Text Index

Create a BM25 inverted index on a node property:

```python
import grafeo

db = grafeo.GrafeoDB()

# Create some nodes with text content
db.execute("INSERT (:Article {title: 'Introduction to Graph Databases'})")
db.execute("INSERT (:Article {title: 'Machine Learning with Python'})")
db.execute("INSERT (:Article {title: 'Graph Neural Networks for NLP'})")

# Create a text index on the title property
text_owner = db.create_index("title", kind="text", label="Article")
```

```typescript
const db = GrafeoDB.create();

await db.execute("INSERT (:Article {title: 'Introduction to Graph Databases'})");
await db.execute("INSERT (:Article {title: 'Machine Learning with Python'})");
await db.execute("INSERT (:Article {title: 'Graph Neural Networks for NLP'})");

const textOwner = await db.createIndex({kind: "text", label: "Article", property: "title"});
```

## Searching

Use `text_search()` to find nodes by keyword relevance:

```python
results = db.text_search("Article", "title", "graph", k=10)
for node_id, score in results:
    print(f"Node {node_id}: score={score:.4f}")
```

```typescript
const results = await db.textSearch("Article", "title", "graph", 10);
for (const [nodeId, score] of results) {
  console.log(`Node ${nodeId}: score ${score}`);
}
```

### Return value semantics

`text_search()` returns a list of `(node_id, score)` tuples sorted by **descending** relevance (higher score = more relevant). BM25 scores are unbounded positive floats whose magnitude depends on corpus statistics, so compare them only within a single query's results.

## In-Query Text Scoring

BM25 is also callable from GQL/Cypher as `text_score()` and `text_match()`,
which the planner pushes down into a `TextScanOperator` when a text index
exists. Two scan modes:

- **Top-K** (`ORDER BY text_score(...) DESC LIMIT k`): returns the `k`
  highest-scoring documents, using the inverted index to avoid scanning
  non-matching docs.
- **Threshold** (`WHERE text_score(...) > t`): returns every document whose
  BM25 score exceeds `t`.

```gql
-- Top-K
MATCH (a:Article)
RETURN a.title, text_score(a.title, 'graph') AS score
ORDER BY text_score(a.title, 'graph') DESC
LIMIT 5

-- Threshold
MATCH (a:Article)
WHERE text_score(a.title, 'graph') > 1.0
RETURN a.title
```

See [filter-expression hybrid search](filter-expressions.md) for the full
syntax, AND/OR composition with vector predicates, and index-missing fallback
behavior.

## Historical search and retention

Epoch-qualified Text search, scoring and corpus statistics return `Result`.
An epoch below the index's `retained_from()` cutoff is an error, not an empty
match or a score computed from today's corpus. Empty queries and zero limits
also check that cutoff. Current-latest search remains available after collection.
Text5 container images and Snapshot12 copies preserve the cutoff and retained
score history. See [retained history](../temporal.md#history-is-subject-to-retention)
for the limits of historical reads and manual `db.gc()?`.

## Auto-Sync Behavior

Text indexes are **automatically maintained** as nodes change. You do not need to rebuild after normal write operations:

```python
# Reuse text_owner created above; normal writes maintain it automatically.

# New nodes are auto-indexed
db.execute("INSERT (:Article {key: 'rust', title: 'Rust Systems Programming'})")

# Updated properties are auto-reindexed
db.execute("MATCH (a:Article {key: 'rust'}) SET a.title = 'Updated Title'")

# Deleted nodes are auto-removed from the index
db.execute("MATCH (a:Article {key: 'rust'}) DELETE a")

# All of the above are reflected in search results immediately,
# no rebuild needed.
```

## Explicit Maintenance

Rebuild is optional maintenance, not a prerequisite after supported writes or
imports. It atomically replaces index contents while retaining the same owner,
resolved configuration, and supported historical visibility.

```python
db.rebuild_index(text_owner)
assert db.drop_index(text_owner)
# Rebuild now raises: a retired owner cannot be resurrected.
text_owner = db.create_index("title", kind="text", label="Article")
```

The recreated index has a new owner ID. Do not mutate the raw store to bypass
normal index maintenance.

## BM25 Configuration

Text indexes use the default BM25 configuration (k1=1.2, b=0.75) with Unicode-aware tokenization and English stop word removal. Custom BM25 parameters are not currently configurable through the API.

The built-in tokenizer's minimum token length defaults to 2, measured in UTF-8
bytes after lowercasing. Set it explicitly through GQL (including from
Python/Node `execute`):

```gql
CREATE INDEX article_words FOR (a:Article) ON (a.title)
USING TEXT {min_token_length: 3}
```

Create this instead of the earlier index on the same label/property. In Rust,
set `CreateIndexRequest.kind` to
`IndexCreateKind::Text { min_token_length: Some(3) }`; `None` selects 2.
Zero is valid. Negative, non-integer and non-Text DDL options are rejected.
The same tokenizer is used for indexing and queries, including after rebuild,
recovery, compaction and snapshot import. This configures the built-in tokenizer;
it does not make arbitrary tokenizer trait objects persistable.
