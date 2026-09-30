# grafeo-c

C FFI bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Building

```bash
# From the Grafeo repository root:
cargo build --release -p grafeo-c --features full

# Output:
#   target/release/libgrafeo_c.so      (Linux)
#   target/release/libgrafeo_c.dylib   (macOS)
#   target/release/grafeo_c.dll        (Windows)
```

The header file is at `crates/bindings/c/grafeo.h`.

The `lpg`, `embedded` (default), `edge`, `native`, and `compact-store` profiles expose the
LPG CRUD, transaction, index-owner and administration entry points. For a
parser-free LPG/RDF build, use `--no-default-features --features native`;
query entry points return errors when their language is unavailable. Optional
language-specific symbols and Text/Vector support still depend on their features.
The memory-only `edge` profile returns `ErrorStorage` for save, backup and restore;
add `storage` to enable those operations. Bare `compact-store` enables direct LPG
operations and compact maintenance without parsers or a storage backend; its
save/backup/restore calls also return `ErrorStorage`.

The `rdf` profile exposes RDF quad and transaction operations, shared
`grafeo_info`, and `grafeo_save` with its storage backend. Open an explicit RDF
database with `grafeo_open_memory_model(GRAFEO_GRAPH_MODEL_RDF)`. Bare
`triple-store` supports those in-memory RDF operations without query parsers;
`grafeo_save` returns `ErrorStorage` until a storage capability is enabled.
Streaming requires both GQL and LPG support; otherwise the streaming symbols
remain available and return an unsupported-query error.

## Quick Start

```c
#include "grafeo.h"
#include <stdio.h>

int main(void) {
    /* Open an in-memory database (returns NULL on error) */
    GrafeoDatabase *db = grafeo_open_memory();
    if (!db) {
        fprintf(stderr, "Error: %s\n", grafeo_last_error());
        return 1;
    }

    /* Create nodes with labels (JSON array) and properties (JSON object) */
    uint64_t alix = grafeo_create_node(db, "[\"Person\"]", "{\"name\":\"Alix\",\"age\":30}");
    uint64_t gus  = grafeo_create_node(db, "[\"Person\"]", "{\"name\":\"Gus\",\"age\":25}");

    /* Create an edge */
    grafeo_create_edge(db, alix, gus, "KNOWS", "{\"since\":2020}");

    /* Query with GQL */
    GrafeoResult *r = grafeo_execute(db, "MATCH (p:Person) RETURN p.name, p.age");
    if (r) {
        printf("Rows: %zu\n", grafeo_result_row_count(r));
        printf("JSON: %s\n", grafeo_result_json(r));
        grafeo_free_result(r);
    } else {
        fprintf(stderr, "Query error: %s\n", grafeo_last_error());
    }

    /* Cleanup */
    grafeo_close(db);
    grafeo_free_database(db);
    return 0;
}
```

Compile with:

```bash
gcc -o example example.c -lgrafeo_c -L/path/to/target/release
```

## API Overview

### Lifecycle

```c
GrafeoDatabase* grafeo_open_memory(void);                   /* in-memory */
GrafeoDatabase* grafeo_open(const char* path);              /* persistent */
GrafeoDatabase* grafeo_open_read_only(const char* path);    /* read-only */
GrafeoDatabase* grafeo_open_single_file(const char* path);  /* single .grafeo file */
GrafeoStatus    grafeo_close(GrafeoDatabase* db);           /* flush and close */
void            grafeo_free_database(GrafeoDatabase* db);   /* free handle */
const char*     grafeo_version(void);                       /* library version */
```

### Query Execution

All query functions return `GrafeoResult*`, or `NULL` on error.

```c
GrafeoResult* grafeo_execute(db, query);                          /* GQL */
GrafeoResult* grafeo_execute_with_params(db, query, params_json); /* GQL + params */
GrafeoResult* grafeo_execute_cypher(db, query);                   /* Cypher */
GrafeoResult* grafeo_execute_gremlin(db, query);                  /* Gremlin */
GrafeoResult* grafeo_execute_graphql(db, query);                  /* GraphQL */
GrafeoResult* grafeo_execute_sparql(db, query);                   /* SPARQL */
GrafeoResult* grafeo_execute_sql(db, query);                      /* SQL/PGQ */
GrafeoResult* grafeo_execute_language(db, language, query, params_json);  /* any language */
```

Each language also has a `_with_params` variant (e.g. `grafeo_execute_cypher_with_params`).

#### Query controls and bounded execution

The complete declarations are in [`grafeo.h`](grafeo.h). A control's deadline
starts when it is created: pass `-1` for no deadline or a nonnegative number of
milliseconds. A control is consumed by one execution. Cancellation handles are
independently owned and may be used by another thread while the execution is
running.

```c
GrafeoQueryControl *control = grafeo_query_control_create(5000);
GrafeoCancelHandle *cancel = grafeo_query_control_cancel_handle(control);
GrafeoQueryOptions options = {
    .control = control, .max_rows = 1000, .max_bytes = 1024 * 1024,
    .language = NULL,
};
GrafeoResult *result = grafeo_execute_with_options(
    db, "MATCH (n) RETURN n", NULL, &options);
if (!result) fprintf(stderr, "code=%s error=%s\n",
                     grafeo_last_error_code(), grafeo_last_error());
else grafeo_free_result(result);

/* A separately owned handle can cancel the execution from another thread. */
if (cancel) grafeo_cancel(cancel);       /* returns GrafeoStatus */
grafeo_cancel_handle_free(cancel);
grafeo_query_control_free(control);
```

Pass `NULL` for options to use defaults. Explicit zero limits are real zero
limits. The same options are accepted by
`grafeo_transaction_execute_with_options`. A cancelled, expired, or resource
limited call returns `NULL` or a non-`GRAFEO_OK` status; inspect
`grafeo_last_error_code()` and `grafeo_last_error()`.

#### Bounded streaming

```c
GrafeoQueryControl *control = grafeo_query_control_create(-1);
GrafeoQueryOptions options = {
    .control = control, .max_rows = 10000, .max_bytes = 4 * 1024 * 1024,
    .language = NULL,
};
GrafeoStream *stream = grafeo_stream_open_with_options(
    db, "MATCH (n) RETURN n", NULL, &options);
GrafeoResult *chunk = NULL;
for (;;) {
    GrafeoStatus status = grafeo_stream_next_chunk(stream, 256, &chunk);
    if (status != GRAFEO_OK) {
        fprintf(stderr, "stream code=%s error=%s\n",
                grafeo_last_error_code(), grafeo_last_error());
        break;
    }
    if (!chunk) break;                    /* clean EOF */
    /* Consume or copy the chunk before releasing it. */
    grafeo_free_result(chunk);
    chunk = NULL;
}
GrafeoStatus closed = grafeo_stream_close(stream);
if (closed != GRAFEO_OK)
    fprintf(stderr, "close code=%s error=%s\n",
            grafeo_last_error_code(), grafeo_last_error());
grafeo_query_control_free(control);
grafeo_stream_free(stream);
```

`max_rows` is the total stream row cap; each chunk request is also capped
internally. `max_bytes` bounds each copied row/chunk. With `NULL` options,
streaming has no total row cap and uses the default per-copy byte limit. Close
is fallible and idempotent, and may interrupt a concurrent pull. Cancellation
and close are safe across threads, but the same pointer must remain live until
all calls using it finish; never free it concurrently with a call.

### Results

```c
const char* grafeo_result_json(const GrafeoResult* r);              /* JSON rows */
size_t      grafeo_result_row_count(const GrafeoResult* r);         /* row count */
double      grafeo_result_execution_time_ms(const GrafeoResult* r); /* timing */
uint64_t    grafeo_result_rows_scanned(const GrafeoResult* r);      /* rows scanned */
const char* grafeo_result_nodes_json(const GrafeoResult* r);        /* extracted nodes */
const char* grafeo_result_edges_json(const GrafeoResult* r);        /* extracted edges */
void        grafeo_free_result(GrafeoResult* r);
```

All `const char*` pointers are valid until the parent `GrafeoResult` is freed.

### Node & Edge CRUD

```c
uint64_t     grafeo_create_node(db, labels_json, properties_json);
uint64_t     grafeo_create_edge(db, source_id, target_id, edge_type, properties_json);
GrafeoStatus grafeo_get_node(db, id, &node);
GrafeoStatus grafeo_get_edge(db, id, &edge);
int32_t      grafeo_delete_node(db, id);
int32_t      grafeo_delete_edge(db, id);
GrafeoStatus grafeo_set_node_property(db, id, key, value_json);
GrafeoStatus grafeo_set_edge_property(db, id, key, value_json);
int32_t      grafeo_remove_node_property(db, id, key);
int32_t      grafeo_remove_edge_property(db, id, key);
int32_t      grafeo_add_node_label(db, id, label);
int32_t      grafeo_remove_node_label(db, id, label);
char*        grafeo_get_node_labels(db, id);  /* free with grafeo_free_string */
```

### Transactions

```c
GrafeoTransaction* tx = grafeo_begin_transaction(db);
GrafeoResult*      r  = grafeo_transaction_execute(tx, "INSERT (:Person {name: 'Alix'})");
grafeo_free_result(r);
grafeo_commit(tx);          /* or grafeo_rollback(tx) */
grafeo_free_transaction(tx);
```

Also available: `grafeo_begin_transaction_with_isolation`, `grafeo_transaction_execute_with_params`, and `grafeo_transaction_execute_language`.

### Vector Search

```c
GrafeoIndexRequest request = {
    .kind = GRAFEO_INDEX_VECTOR,
    .options = GRAFEO_INDEX_LABEL_PRESENT | GRAFEO_INDEX_DIMENSIONS_PRESENT,
    .label = {(const uint8_t*)"Document", 8},
    .property = {(const uint8_t*)"embedding", 9},
    .dimensions = 384
};
uint32_t owner;
GrafeoStatus status = grafeo_create_index(db, &request, &owner);
/* Check status before using owner; report grafeo_last_error() on failure. */

uint64_t *ids = NULL;
float *distances = NULL;
size_t count = 0;
grafeo_vector_search(db, "Document", "embedding", query_vec, 384, 5, -1, &ids, &distances, &count);
grafeo_free_vector_results(ids, distances, count);
```

Create Property, BTree, Text, or Vector indexes with `grafeo_create_index`.
`grafeo_rebuild_index(db, owner)` retains that owner's resolved configuration;
missing owners are errors. `grafeo_drop_index(db, owner, &dropped)` returns a
status and writes 1 for removal or 0 for absence. Every error remains an error,
not a false drop result. Read/search APIs remain available behind their features.

Requests use UTF-8 pointer-and-byte-length spans, not NUL-terminated strings.
`graph_count = 0` selects root; each graph span is one literal component.
A single empty span selects an empty-named graph, not root. Optional fields
use explicit presence bits, including empty strings and zero numeric values;
invalid or irrelevant options are rejected. Memory remains caller-owned for
the duration of the call. Also available: `grafeo_mmr_search` and
`grafeo_batch_create_nodes`.

### Error Handling

Functions that return pointers use `NULL` for errors. Functions that return `GrafeoStatus` use `GRAFEO_OK` (0) for success. In both cases, call `grafeo_last_error()` for details:

```c
GrafeoResult *r = grafeo_execute(db, query);
if (!r) {
    fprintf(stderr, "Error: %s\n", grafeo_last_error());
    grafeo_clear_error();
}
```

### Memory Management

- Opaque pointers (`GrafeoDatabase*`, `GrafeoResult*`, etc.) must be freed with their `grafeo_free_*` function
- `const char*` from accessor functions (e.g. `grafeo_result_json`, `grafeo_edge_type`) are valid until the parent is freed: do NOT free them
- `char*` from functions like `grafeo_info` and `grafeo_get_node_labels` are caller-owned: free with `grafeo_free_string`

## Features

- GQL, Cypher, SPARQL, Gremlin, GraphQL, and SQL/PGQ query languages
- Full node/edge CRUD with JSON property serialization
- ACID transactions with configurable isolation levels
- HNSW vector similarity search with batch operations and MMR
- Property indexes for fast lookups
- Schema context for multi-tenant graphs
- Change data capture (CDC)
- Thread-safe for concurrent use

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Go Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go) (uses this library via CGO)
- [C# Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/csharp) (uses this library via P/Invoke)
- [Dart Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/dart) (uses this library via dart:ffi)

## License

Apache-2.0


### Bounded change pages

Build with `cdc` (included in the default `embedded` profile), then enable capture
with `grafeo_set_cdc_enabled(db, true)`. `grafeo_changes_after` reads the shared
feed; `grafeo_node_history_after` and `grafeo_edge_history_after` additionally
select an entity and inclusive minimum epoch. These readers use the native
Session authorization path.

Pass `(NULL, 0)` for the first cursor, then the exact 97 bytes returned by
`grafeo_change_page_cursor`. Row and byte limits are required and positive;
bytes count native event encodings, excluding the JSON/page envelope. An empty
page can advance; an unchanged cursor means EOF. Copy cursor bytes before
freeing a page if they will be used later.

```c
GrafeoChangePage* page = grafeo_changes_after(db, NULL, 0, 1, 4096);
if (page != NULL) {
    puts(grafeo_change_page_events_json(page));
    grafeo_free_change_page(page);
}
```

The page owns its JSON and cursor independently of the database. Accessor
pointers are borrowed until `grafeo_free_change_page`; never free them separately.
Pages can be freed immediately on early stop. JSON coordinates are exact decimal
strings, including entity IDs, epochs, HLC timestamps, graph incarnations and
edge endpoints. Events retain node labels, edge type/endpoints and RDF terms.

A null result is an error: read `grafeo_last_error_code` on the same thread.
Invalid, foreign and evicted cursors use `GRAFEO-S004`, `GRAFEO-S005` and
`GRAFEO-S006`; an oversized first event uses `GRAFEO-S001`. In-memory cursors last
for that database lifetime; persistent directory stores support native reopen.
