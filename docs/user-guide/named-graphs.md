---
title: Named Graphs
description: Working with multiple named graphs in a single Grafeo database.
tags:
  - graphs
  - multi-graph
---

# Named Graphs

Grafeo supports multiple named graphs within a single database instance. Each graph has its own nodes, edges, labels, indexes and MVCC versioning. This lets you partition data into logical units while sharing a single database connection and benefiting from cross-graph transactional guarantees.

Every database starts with an implicit **default graph**. All queries target the default graph until you explicitly create and switch to a named graph.

## Creating Named Graphs

Use `CREATE GRAPH` to add a new named graph. The basic form creates an open (schema-free) graph:

```sql
CREATE GRAPH friends
```

### Variants

| Syntax | Description |
|--------|-------------|
| `CREATE GRAPH g` | Create an open (schema-free) graph |
| `CREATE GRAPH g ANY` | Explicitly open: any node/edge types allowed |
| `CREATE GRAPH g OPEN` | Synonym for `ANY` |
| `CREATE GRAPH g IF NOT EXISTS` | No-op if the graph already exists |
| `CREATE GRAPH g TYPED my_type` | Bind the graph to a declared graph type |
| `CREATE GRAPH g2 LIKE g1` | Copy the schema (graph type) of `g1`, without data |
| `CREATE GRAPH g3 AS COPY OF g1` | Copy both schema and data from `g1` |

### Examples

```sql
-- Open graph (no type constraints)
CREATE GRAPH friends

-- Idempotent creation
CREATE GRAPH friends IF NOT EXISTS

-- Copy structure only
CREATE GRAPH friends_staging LIKE friends

-- Full clone with data
CREATE GRAPH friends_backup AS COPY OF friends

-- Typed graph (requires a previously created graph type)
CREATE GRAPH TYPE SocialNetwork (
    NODE TYPE Person (name STRING NOT NULL),
    EDGE TYPE KNOWS (since INT64)
)
CREATE GRAPH social TYPED SocialNetwork
```

### Python API

```python
import grafeo

db = grafeo.GrafeoDB()
db.create_graph("friends")
```

### Rust API

```rust
use grafeo::GrafeoDB;

let db = GrafeoDB::new_in_memory();
db.create_graph("friends")?;
```

## Switching Graphs

Once a named graph exists, switch to it with `USE GRAPH` or `SESSION SET GRAPH`. Both set the active graph for the current session; `SESSION SET GRAPH` follows the ISO/IEC 39075 session-state grammar.

```sql
-- Switch to the friends graph
USE GRAPH friends

-- ISO session syntax (equivalent)
SESSION SET GRAPH friends

-- Return to the default graph
USE GRAPH DEFAULT
```

After switching, all subsequent queries in that session read from and write to the selected graph.

### Python API

```python
db.set_current_graph("friends")

# Check which graph is active
print(db.current_graph())  # "friends"
```

### Rust API

```rust
use grafeo::GraphPath;

let session = db.session();
session.execute("CREATE GRAPH friends")?;
session.execute("USE GRAPH friends")?;

assert_eq!(session.current_graph_path(), GraphPath::from_components(&["friends"])?);

// Native selection is absolute, independent of the language's current schema.
session.use_graph_path(&GraphPath::root())?;
```

Native paths are literal component sequences: the root `[]`, `["default"]`,
`["a/b"]`, and `["a", "b"]` are different graphs. Selection requires an existing,
read-authorized graph; failure leaves the current selection unchanged. Changing
the language schema does not retarget an explicitly selected native path.
Language graph/schema names still resolve to one root child, not nested paths.

Constraints follow the selected storage path, not an unrelated language schema.
Existing schema-qualified root children retain their schema's constraints; root
and nested paths use the root constraint namespace.

### Native recursive lifecycle (Rust)

Create and drop exact paths through `GrafeoDB` or an existing `Session`:

```rust
use grafeo::{GrafeoDB, GraphPath};

let db = GrafeoDB::new_in_memory();
let mut session = db.session();
session.begin_transaction()?;
let parent = GraphPath::from_components(&["research"])?;
let child = parent.child("papers/2026")?; // one literal child name
session.create_graph_path(&parent)?;
session.create_graph_path(&child)?;
session.use_graph_path(&child)?;
session.execute("INSERT (:Paper {title: 'Temporal graphs'})")?;
session.commit()?;

db.drop_graph_path(&parent)?; // drops the subtree and its index owners
```

Creation requires an existing parent, including one staged in the same
transaction; it does not create missing ancestors. Existing paths return
`false`. Dropping an absent path returns `false`; root and catalog schema-default
partitions cannot be dropped. Native names are never rewritten: a registered
schema prefix must use its canonical spelling.

Graph lifecycle, data and indexes share transaction rollback, savepoints and
durable publication. A concurrently replaced parent causes a conflict instead
of redirecting its staged child. Explicit native operations check their target's
grants, independently of the current selection. They do not retarget that
selection after DROP: select another existing path before querying or writing.
Recursive topology and nested mutations survive committed WAL recovery,
compaction and current-format whole-database copies.

## Listing and Dropping Graphs

### Listing

`SHOW GRAPHS` returns all named graphs visible in the current schema context:

```sql
SHOW GRAPHS
```

In Python:

```python
names = db.list_graphs()
print(names)  # ["friends", "colleagues"]
```

In Rust:

```rust
let names = db.list_graphs();
println!("{names:?}");
```

### Dropping

Remove a named graph with `DROP GRAPH`. All data in the graph is deleted. If the session is currently using that graph, the session resets to the default graph automatically.

```sql
DROP GRAPH friends

-- No error if the graph does not exist
DROP GRAPH IF EXISTS friends
```

In Python:

```python
db.drop_graph("friends")
```

In Rust:

```rust
db.drop_graph("friends")?;
```

## Cross-Graph Transactions

`USE GRAPH` works within an active transaction. You can switch between graphs mid-transaction, and the commit or rollback applies atomically across all graphs that were touched.

```sql
START TRANSACTION

USE GRAPH friends
INSERT (:Person {name: 'Alix'})

USE GRAPH colleagues
INSERT (:Person {name: 'Gus'})

COMMIT
-- Both inserts succeed or neither does
```

If a conflict is detected in any graph during commit, the entire transaction is
rolled back. Savepoints span graph boundaries and restore transaction-local
topology and data. The current graph/schema selectors survive savepoint rollback;
retained incarnation checks prevent them from silently selecting a replacement.

## Data Isolation

Each named graph has completely separate storage. Queries only see data in the active graph, and mutations in one graph never affect another.

```sql
CREATE GRAPH alpha
CREATE GRAPH beta

USE GRAPH alpha
INSERT (:Person {name: 'Alix'})

USE GRAPH beta
INSERT (:Person {name: 'Gus'})

-- Only Gus is visible here
MATCH (p:Person) RETURN p.name
```

Internally, the query plan cache keys include the active graph name. This means switching graphs does not cause stale cache hits: plans compiled for `alpha` are never reused when the session is using `beta`.

## Persistence

Named graphs are fully persisted when the database uses a persistent storage
backend. `CREATE GRAPH` and `DROP GRAPH` publish the catalog binding and the
physical graph-partition change in one committed WAL transaction, so recovery
cannot expose metadata without its graph (or the graph without its metadata).

```python
import grafeo

# Persistent database
db = grafeo.GrafeoDB("my_data.grafeo")
db.execute("CREATE GRAPH friends")
db.execute("USE GRAPH friends")
db.execute("INSERT (:Person {name: 'Alix'})")

# After restart, the friends graph and its data are still present
db2 = grafeo.GrafeoDB("my_data.grafeo")
db2.execute("USE GRAPH friends")
result = db2.execute("MATCH (p:Person) RETURN p.name")
# Returns Alix
```

Current whole-database snapshots include recursive named graphs, the default
graph and their exact index state.

## Schema-Scoped Graphs

Language schemas are catalog namespaces, not native path components. Create the
schema before selecting it. Namespace-incarnation and grant checks apply to
graph lifecycle; a schema-qualified graph name still resolves to one root child.

Grafeo supports ISO/IEC 39075 session schemas. When a schema is set, graph names are resolved within that schema context. `CREATE GRAPH` creates the graph under the active schema, and `SHOW GRAPHS` only returns graphs belonging to that schema.

```sql
-- Set the session schema
CREATE SCHEMA analytics
SESSION SET SCHEMA analytics

-- This graph is scoped to "analytics"
CREATE GRAPH daily_metrics

-- Only shows graphs in the analytics schema
SHOW GRAPHS

-- Switch schema
CREATE SCHEMA reporting
SESSION SET SCHEMA reporting
CREATE GRAPH monthly_summary
```

Schema and graph are independent session settings (per ISO/IEC 39075 Section 7.1). Changing the schema does not reset the active graph, and changing the graph does not reset the schema.

```sql
-- Reset schema without affecting the active graph
SESSION RESET SCHEMA

-- Reset graph without affecting the schema
SESSION RESET GRAPH

-- Reset everything
SESSION RESET
```
