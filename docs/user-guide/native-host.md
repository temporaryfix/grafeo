---
title: Native host profiles
description: Compose embedded LPG and RDF capabilities from the review source.
---

# Native host profiles

The review source retains one engine with selectable capabilities. These
examples use a local path so a published package is not mistaken for this
unreleased implementation.

## Temporal property-graph host

```toml
[dependencies]
grafeo = { path = "../../crates/grafeo", default-features = false, features = ["temporal-host"] }
```

`temporal-host` groups LPG, compact storage, the opaque statement table, WAL and
text search. Add a query-language feature when needed. This profile name is
provisional; it is a compilation choice within the same engine.

## RDF

```toml
[dependencies]
grafeo = { path = "../../crates/grafeo", default-features = false, features = ["native", "sparql"] }
```

`native` supplies parser-free LPG/RDF and persistence capabilities. Add `sparql`
for RDF queries and `shacl` for supported validation. Select `GraphModel::Rdf`
or `GraphModel::Both` in configuration and use a Session for a transaction that
groups several operations. The opaque statement-table host API is not a
replacement for RDF Session/WAL writes.

## Identity, snapshots and world cuts

The epoch returned by a successful commit identifies a publication in that
store. Retain it with the StoreId. Historical visibility depends on retained
versions; an epoch or WorldCut does not itself pin retention.

A WorldCut describes a state and its format/metadata identity. Integrity checks
establish consistency of the bytes and metadata, not producer authentication.
An application must establish the provenance it needs separately.

Consult [retained history](temporal.md), [transactions](transactions.md), and
[persistence](persistence/index.md) for the distinctions between current reads,
historical views, valid time and recovery. The review guide identifies which
source checks have been run and which release gates remain open.
