---
title: Go API
description: API reference for the Grafeo Go bindings.
---

# Go API

Go bindings for Grafeo via CGO. Requires the `grafeo-c` shared library.

```bash
go get github.com/GrafeoDB/grafeo/crates/bindings/go
```

## Requirements

- Go 1.22+
- CGO enabled (`CGO_ENABLED=1`)
- The `grafeo-c` shared library (`libgrafeo_c.so` / `libgrafeo_c.dylib` / `grafeo_c.dll`)

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

func main() {
    db, err := grafeo.OpenInMemory()
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

    result, err := db.Execute("MATCH (p:Person) RETURN p.name, p.age")
    if err != nil {
        log.Fatal(err)
    }
    for _, row := range result.Rows {
        fmt.Printf("Name: %v, Age: %v\n", row["p.name"], row["p.age"])
    }
}
```

## Database

```go
db, err := grafeo.OpenInMemory()                // in-memory
db, err := grafeo.Open("./path")                // persistent (auto-detects format)
db, err := grafeo.OpenSingleFile("./data.grafeo") // single-file .grafeo format
defer db.Close()

db.NodeCount()   // number of nodes
db.EdgeCount()   // number of edges
```

## Query Languages

```go
result, err := db.Execute(gql)            // GQL (ISO standard)
result, err := db.ExecuteCypher(query)     // Cypher
result, err := db.ExecuteGremlin(query)    // Gremlin
result, err := db.ExecuteGraphQL(query)    // GraphQL
result, err := db.ExecuteSPARQL(query)     // SPARQL
result, err := db.ExecuteSQL(query)        // SQL/PGQ

// Parameterized queries (accepts map[string]any, marshals to JSON internally)
result, err := db.ExecuteParams(
    "MATCH (p:Person) WHERE p.age > $min RETURN p.name",
    map[string]any{"min": 25},
)

// Each language also has a WithParams variant (accepts raw JSON string)
result, err := db.ExecuteCypherWithParams(query, paramsJSON)
result, err := db.ExecuteGremlinWithParams(query, paramsJSON)
result, err := db.ExecuteGraphQLWithParams(query, paramsJSON)
result, err := db.ExecuteSPARQLWithParams(query, paramsJSON)
result, err := db.ExecuteSQLWithParams(query, paramsJSON)

// Generic language dispatch: "gql", "cypher", "gremlin", "graphql", "sparql", "sql"
result, err := db.ExecuteLanguage("cypher", query, paramsJSON)
```

## Node & Edge CRUD

```go
func crudExample(db *grafeo.Database) error {
    node, err := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
    if err != nil {
        return err
    }
    target, err := db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus"})
    if err != nil {
        return err
    }
    edge, err := db.CreateEdge(node.ID, target.ID, "KNOWS", map[string]any{"since": 2024})
    if err != nil {
        return err
    }
    id, eid := node.ID, edge.ID // CRUD methods take IDs, not *Node or *Edge.

    if _, err := db.GetNode(id); err != nil {
        return err
    }
    if _, err := db.GetEdge(eid); err != nil {
        return err
    }
    if err := db.SetNodeProperty(id, "age", 31); err != nil {
        return err
    }
    if err := db.SetEdgeProperty(eid, "weight", 0.5); err != nil {
        return err
    }

    if _, err := db.DeleteEdge(eid); err != nil {
        return err
    }
    if _, err := db.DeleteNode(id); err != nil {
        return err
    }
    _, err = db.DeleteNode(target.ID)
    return err
}
```

Both property setters return `nil` only after a successful write. A missing or
deleted target returns an error; the setter does not create it. Other engine
write failures propagate through the same error return.

## Transactions

```go
tx, err := db.BeginTransaction()
result, err := tx.Execute("INSERT (:Person {name: 'Harm'})")
err = tx.Commit()   // or tx.Rollback()
```

## Index Owners

```go
func (db *Database) CreateIndex(request CreateIndexRequest) (IndexID, error)
func (db *Database) DropIndex(owner IndexID) (bool, error)
func (db *Database) RebuildIndex(owner IndexID) error
```

Creation returns a committed unsigned 32-bit owner ID. Duplicate names or physical targets are errors. Graph paths are component arrays: `[]` selects root, `[""]` an empty-named child, and `["a/b"]` differs from `["a", "b"]`. Property/BTree indexes forbid a label; Text/Vector require one. Rebuild atomically preserves the owner and its full resolved configuration; it does not recreate a dropped index. Drop returns false only for an absent owner. Engine failures propagate through the binding's error channel.

Calls are synchronous. `IndexID` is backed by `uint32`. Requests use
`Graph []string`, `Property string`, and `Kind` (`PropertyIndex`,
`BTreeIndex`, `TextIndex`, or `VectorIndex`). Optional strings
`Name`, `Label`, `Metric`, and `Quantization` are `*string`;
Vector-only `Dimensions`, `M`, and `EfConstruction` are `*uint`.
Pointers distinguish absence from explicit zero/empty values.

Current 0.0.1 limitation: index-owner mutations on WAL-backed databases are rejected. Saving or checkpointing owner-bearing state, including retained owner-ID allocation history after drops, also fails closed until the current persistence formats support those owners. The in-memory examples below are not a persistence guarantee.

## Vector Search

```go
// Create an HNSW owner.
label, metric := "Doc", "cosine"
dimensions, m, efConstruction := uint(384), uint(16), uint(200)
owner, err := db.CreateIndex(grafeo.CreateIndexRequest{
    Kind: grafeo.VectorIndex, Label: &label, Property: "emb",
    Dimensions: &dimensions, Metric: &metric, M: &m,
    EfConstruction: &efConstruction,
})
if err != nil { log.Fatal(err) }
if err := db.RebuildIndex(owner); err != nil { log.Fatal(err) }

// Search
results, err := db.VectorSearch("Doc", "emb", queryVec, 10)

// With options
results, err = db.VectorSearch("Doc", "emb", queryVec, 10,
    grafeo.WithEf(100))

// MMR search
results, err = db.MmrSearch("Doc", "emb", queryVec, 5, -1, 0.5, -1)
```

## Building the Shared Library

```bash
cargo build --release -p grafeo-c --features full
```

## Links

- [pkg.go.dev](https://pkg.go.dev/github.com/GrafeoDB/grafeo/crates/bindings/go)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go)
