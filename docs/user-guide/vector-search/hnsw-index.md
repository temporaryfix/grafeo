---
title: HNSW Index
description: Hierarchical Navigable Small World index for fast approximate nearest neighbor search.
tags:
  - vector-search
  - hnsw
  - index
---

# HNSW Index

The HNSW (Hierarchical Navigable Small World) index provides O(log n) approximate nearest neighbor search, making vector queries fast even with millions of vectors.


Current 0.0.1 limitation: index-owner mutations on WAL-backed databases are rejected. Saving or checkpointing owner-bearing state, including retained owner-ID allocation history after drops, also fails closed until the current persistence formats support those owners. The in-memory examples below are not a persistence guarantee.

## Overview

HNSW builds a multi-layer graph structure where:

- Higher layers have fewer nodes for fast global navigation
- Lower layers have more connections for accurate local search
- Search starts at the top layer and descends to find nearest neighbors

## Creating an HNSW Index

### GQL Syntax

```sql
-- Create a vector index on the embedding property
CREATE VECTOR INDEX movie_embeddings ON :Movie(embedding)
  DIMENSION 384 METRIC 'cosine'

-- Dimensions and metric are optional (defaults: auto-detect dimensions, cosine)
CREATE VECTOR INDEX document_embeddings ON :Document(embedding)
  DIMENSION 768 METRIC 'euclidean'
```

### Python API

```python
import grafeo

db = grafeo.GrafeoDB()

# Create index with default settings
movie_owner = db.create_index(
    "embedding",         # property
    kind="vector",
    label="Movie",
    dimensions=384,      # optional
    metric="cosine"      # optional (default: cosine)
)

# With HNSW tuning parameters
document_owner = db.create_index(
    "embedding", kind="vector", label="Document",
    dimensions=768,
    metric="euclidean",
    m=32,                # connections per node (default: 16)
    ef_construction=256, # build-time quality (default: 128)
    ef=128               # default search depth (default: 50)
)
```

## Configuration Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `dimensions` | Required | Number of dimensions in vectors |
| `metric` | `cosine` | Distance metric: `cosine`, `euclidean`, `dot_product`, `manhattan` |
| `m` | 16 | Max connections per node (higher = better recall, more memory) |
| `ef_construction` | 128 | Build-time beam width (higher = better index quality, slower build) |
| `ef` | 50 | Search-time beam width (higher = better recall, slower search) |

## Tuning Parameters

### M (Connections per Node)

Controls the graph connectivity:

- **Lower (8-12)**: Faster build, less memory, lower recall
- **Default (16)**: Good balance for most use cases
- **Higher (24-48)**: Better recall, more memory, slower build

Configure with Python `create_index(..., kind="vector")` options or the Node `createIndex({kind: "vector", ...})` request.

### ef_construction (Build Quality)

Controls index build quality:

- **Lower (64)**: Faster build, slightly lower quality
- **Default (128)**: Good balance
- **Higher (256-512)**: Best quality, slower build

Configure with Python `create_index(..., kind="vector")` options or the Node `createIndex({kind: "vector", ...})` request.

### ef (Search Quality)

Set a positive index-level search depth with Python `create_index(..., ef=128)`
or GQL:

```sql
CREATE INDEX document_vectors FOR (d:Document) ON (d.embedding)
USING VECTOR {dimensions: 768, metric: 'euclidean', ef: 128}
```

The resolved value survives rebuild and reopen. The GQL search procedure uses
this index configuration:

```sql
CALL grafeo.search.vector('Document', 'embedding', $query, 10)
```

Python can override the depth for one call without changing the index:

```python
results = db.vector_search("Document", "embedding", embedding, k=10, ef=200)
```

Higher depth trades more search work for recall. The effective beam is at least
`k`; visibility and property filtering may widen it. An ordinary scalar
`ORDER BY` distance query can use an exact scan and does not by itself prove
that an HNSW index was used.

## Performance Characteristics

### Time Complexity

| Operation | Complexity |
| --------- | ---------- |
| Build | O(n log n) |
| Insert | O(log n) |
| Search | O(log n) |
| Delete | O(log n) |

### Memory Usage

Approximate memory per vector:

```
memory = dimensions * 4 bytes (f32)
       + m * 2 * 8 bytes (connections)
       + overhead (~50 bytes)
```

For 1 million 384-dimensional vectors with m=16:

```
1M * (384 * 4 + 16 * 2 * 8 + 50) ≈ 1.8 GB
```

## Recall vs Speed Tradeoffs

| Configuration | Recall | Search Time | Memory |
| ------------- | ------ | ----------- | ------ |
| m=8, ef=25 | ~85% | Fastest | Lowest |
| m=16, ef=50 (default) | ~95% | Fast | Medium |
| m=32, ef=100 | ~99% | Moderate | Higher |
| m=48, ef=200 | ~99.5% | Slower | Highest |

## Best Practices

### 1. Choose the Right Metric

```sql
-- For text embeddings (usually normalized)
CREATE VECTOR INDEX idx ON :Doc(embedding) METRIC 'cosine'

-- For image embeddings or spatial data
CREATE VECTOR INDEX idx ON :Doc(embedding) METRIC 'euclidean'

-- For retrieval models with dot product training
CREATE VECTOR INDEX idx ON :Doc(embedding) METRIC 'dot_product'
```

### 2. Build Index After Bulk Loading

```python
# Load data first
for doc in documents:
    db.execute("INSERT (:Document {title: $title, embedding: $emb})",
               {"title": doc.title, "emb": doc.embedding})

# Then create index (faster than incremental inserts)
vector_owner = db.create_index("embedding", kind="vector", label="Document", dimensions=384, metric="cosine")
```

### 3. Monitor Recall

Periodically verify recall against brute-force search:

```python
# Sample query
query = get_random_query()

# HNSW result
hnsw_result = db.execute("MATCH (d:Document) RETURN d.id ORDER BY cosine_similarity(d.embedding, $q) DESC LIMIT 10",
                          {"q": query})

# Compare with brute-force (on small sample)
# to ensure HNSW recall meets requirements
```

## Combining with Quantization

For memory-constrained environments, combine HNSW with quantization:

See the [Quantization](quantization.md) page for how to use scalar, binary and product quantization
to reduce memory usage while maintaining search quality.

See [Quantization](quantization.md) for details.

## Next Steps

- [**Quantization**](quantization.md) - Reduce memory with vector compression
- [**Python API**](python-api.md) - Low-level Python bindings
