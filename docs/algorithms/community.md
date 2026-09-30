---
title: Community Detection
description: Community detection algorithms.
tags:
  - algorithms
  - community
---

# Community Detection

Find clusters and communities within graphs.

## Louvain Algorithm

Runs local moving and community aggregation repeatedly, following [Blondel et al. (2008)](https://arxiv.org/abs/0803.0476). Reported modularity includes the full community degree term. The two-level ring-of-triangles control improves from Q=0.650 to Q=0.675 after aggregation.

```python
import grafeo

db = grafeo.GrafeoDB()
algs = db.algorithms

result = algs.louvain()
for node_id, community_id in result["communities"].items():
    print(node_id, community_id)
print(result["modularity"])
```

## Label Propagation

Uses synchronous [LDBC Graphalytics CDLP](https://ldbcouncil.org/ldbc_graphalytics_docs/graphalytics_spec.pdf) updates: each iteration reads the previous labels, ties choose the smallest label, and isolated nodes retain their label. Reciprocal edges contribute in both directions. Returned community IDs are normalized; they describe partitions rather than raw LDBC vertex labels. Bipartite labels may oscillate until the iteration limit.

```python
algs = db.algorithms
communities = algs.label_propagation()
```

## Connected Components

Find disconnected subgraphs.

```python
algs = db.algorithms
components = algs.connected_components()

print(f"Found {len(components)} components")
for i, comp in enumerate(components):
    print(f"Component {i}: {len(comp)} nodes")
```

## Strongly Connected Components

For directed graphs.

```python
algs = db.algorithms
sccs = algs.strongly_connected_components()
```

## Weakly Connected Components

For directed graphs, ignoring edge direction.

```python
algs = db.algorithms
wccs = algs.weakly_connected_components()
```

## Triangle Count

Count triangles for clustering analysis.

```python
algs = db.algorithms
triangles = algs.triangles()
print(f"Total triangles: {triangles}")
```

## Stochastic Block Partition

Bayesian community detection using the degree-corrected stochastic block model. Minimizes the description length (MDL) of the graph under the generative model, making it more principled than modularity-based methods for graphs with heterogeneous degree distributions.

```python
algs = db.algorithms

# Auto-detect optimal number of blocks
result = algs.stochastic_block_partition()
print(f"Found {result['num_blocks']} blocks (DL: {result['description_length']:.2f})")

# Target a specific number of blocks
result = algs.stochastic_block_partition(num_blocks=4)

# Incremental update after adding edges (warm start from prior partition)
result2 = algs.stochastic_block_partition(prior_partition=result['partition'])
```

Part of the [GraphChallenge](https://graphchallenge.mit.edu/) streaming benchmark suite.

## Partition Quality Metrics

Compare two community assignments (e.g., predicted vs ground truth):

```python
from grafeo.algorithms import metrics

ri  = metrics.rand_index(partition_a, partition_b)
ari = metrics.adjusted_rand_index(partition_a, partition_b)
nmi = metrics.normalized_mutual_information(partition_a, partition_b)

prec = metrics.pairwise_precision(predicted, truth)
rec  = metrics.pairwise_recall(predicted, truth)
```

| Metric | Range | 1.0 means |
|--------|-------|-----------|
| Rand Index | [0, 1] | Identical partitions |
| Adjusted Rand Index | [-1, 1] | Identical (0 = random) |
| Normalized Mutual Information | [0, 1] | Identical partitions |
| Pairwise Precision | [0, 1] | Every predicted co-cluster is true |
| Pairwise Recall | [0, 1] | Every true co-cluster is predicted |

## Use Cases

| Algorithm | Best For |
|-----------|----------|
| Louvain | Large graphs, quality clusters |
| Label Propagation | Fast, scalable |
| Stochastic Block Partition | Heterogeneous graphs, Bayesian inference |
| Connected Components | Graph structure analysis |
| Triangle Count | Clustering coefficient |
