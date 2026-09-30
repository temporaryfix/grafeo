/* Grafeo C API
 *
 * Link against libgrafeo_c.so (Linux), libgrafeo_c.dylib (macOS),
 * or grafeo_c.dll (Windows).
 *
 * Memory management:
 *   - Opaque pointers must be freed with their grafeo_free_* function.
 *   - Strings documented as "free with grafeo_free_string" are caller-owned.
 *   - Pointers documented as "valid until free" must NOT be freed separately.
 *
 * Error handling:
 *   - Functions return GrafeoStatus (0 = success).
 *   - On error, call grafeo_last_error() for a human-readable message.
 */

#ifndef GRAFEO_H
#define GRAFEO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes -------------------------------------------------------- */

typedef enum {
    GRAFEO_OK                  = 0,
    GRAFEO_ERROR_DATABASE      = 1,
    GRAFEO_ERROR_QUERY         = 2,
    GRAFEO_ERROR_TRANSACTION   = 3,
    GRAFEO_ERROR_STORAGE       = 4,
    GRAFEO_ERROR_IO            = 5,
    GRAFEO_ERROR_SERIALIZATION = 6,
    GRAFEO_ERROR_INTERNAL      = 7,
    GRAFEO_ERROR_NULL_POINTER  = 8,
    GRAFEO_ERROR_INVALID_UTF8  = 9,
    GRAFEO_ERROR_CANCELLED     = 10,
    GRAFEO_ERROR_DEADLINE      = 11,
    GRAFEO_ERROR_RESOURCE_LIMIT = 12
} GrafeoStatus;

/* ---- Transaction isolation levels ---------------------------------------- */

typedef enum {
    GRAFEO_ISOLATION_READ_COMMITTED = 0,
    GRAFEO_ISOLATION_SNAPSHOT       = 1,
    GRAFEO_ISOLATION_SERIALIZABLE   = 2
} GrafeoIsolationLevel;

/* ---- Graph model (chosen at create; stored on the file) ------------------ */
/* ABI is uint8_t (Rust u8). Do not use a C enum here: enums are int-sized. */

#define GRAFEO_GRAPH_MODEL_LPG  ((uint8_t)0)
#define GRAFEO_GRAPH_MODEL_RDF  ((uint8_t)1)
#define GRAFEO_GRAPH_MODEL_BOTH ((uint8_t)2)

/* ---- Opaque types -------------------------------------------------------- */

typedef struct GrafeoDatabase    GrafeoDatabase;
typedef struct GrafeoTransaction GrafeoTransaction;
typedef struct GrafeoResult      GrafeoResult;
typedef struct GrafeoChangePage  GrafeoChangePage;
typedef struct GrafeoNode        GrafeoNode;
typedef struct GrafeoEdge        GrafeoEdge;
typedef struct GrafeoStream      GrafeoStream;
typedef struct GrafeoQueryControl GrafeoQueryControl;
typedef struct GrafeoCancelHandle GrafeoCancelHandle;

/* NULL options uses defaults. Explicit zero limits are real zero limits.
 * A control is consumed by one execution; NULL control creates a fresh owner.
 * language may be NULL for the default language. */
typedef struct {
    GrafeoQueryControl* control;
    size_t max_rows;
    size_t max_bytes;
    const char* language;
} GrafeoQueryOptions;

/* timeout_ms: -1 for no deadline; otherwise milliseconds from creation.
 * Clone cancellation handles for independently freed owners. Each allocation
 * must remain live throughout its calls and be freed exactly once; cancel may
 * run concurrently with execution. Free never races uses of the same pointer. */
GrafeoQueryControl* grafeo_query_control_create(int64_t timeout_ms);
GrafeoCancelHandle* grafeo_query_control_cancel_handle(const GrafeoQueryControl* control);
GrafeoCancelHandle* grafeo_cancel_handle_clone(const GrafeoCancelHandle* handle);
GrafeoStatus grafeo_cancel(const GrafeoCancelHandle* handle);
void grafeo_cancel_handle_free(GrafeoCancelHandle* handle);
void grafeo_query_control_free(GrafeoQueryControl* control);
GrafeoResult* grafeo_execute_with_options(GrafeoDatabase* db, const char* query,
    const char* params_json, const GrafeoQueryOptions* options);
GrafeoResult* grafeo_transaction_execute_with_options(GrafeoTransaction* tx,
    const char* query, const char* params_json, const GrafeoQueryOptions* options);
const char* grafeo_last_error_code(void);

/* ---- Error handling ------------------------------------------------------ */

const char* grafeo_last_error(void);
void        grafeo_clear_error(void);

/* ---- Lifecycle ----------------------------------------------------------- */

GrafeoDatabase* grafeo_open_memory(void);
GrafeoDatabase* grafeo_open_memory_model(uint8_t model);
GrafeoDatabase* grafeo_open(const char* path);
GrafeoDatabase* grafeo_open_with_model(const char* path, uint8_t model);
GrafeoDatabase* grafeo_open_read_only(const char* path);
GrafeoDatabase* grafeo_open_single_file(const char* path);
GrafeoStatus    grafeo_close(GrafeoDatabase* db);
void            grafeo_free_database(GrafeoDatabase* db);
const char*     grafeo_version(void);
uint8_t         grafeo_graph_model(const GrafeoDatabase* db);

/* ---- Change Data Capture ------------------------------------------------- */

void grafeo_set_cdc_enabled(GrafeoDatabase* db, bool enabled);
bool grafeo_is_cdc_enabled(GrafeoDatabase* db);

/* CDC feature required. NULL/0 cursor starts at the retained floor; otherwise
 * exactly GRAFEO_CDC_CURSOR_LEN readable bytes are required. Limits are positive;
 * max_bytes counts native event encodings, excluding JSON/page envelopes.
 * Returns NULL on error; read the structured error on the same thread.
 * A page owns its events/cursor independently of the database until freed. */
#define GRAFEO_CDC_CURSOR_LEN 97
GrafeoChangePage* grafeo_changes_after(GrafeoDatabase* db, const uint8_t* cursor,
    size_t cursor_len, size_t max_events, size_t max_bytes);
GrafeoChangePage* grafeo_node_history_after(GrafeoDatabase* db, uint64_t node_id,
    uint64_t since_epoch, const uint8_t* cursor, size_t cursor_len,
    size_t max_events, size_t max_bytes);
GrafeoChangePage* grafeo_edge_history_after(GrafeoDatabase* db, uint64_t edge_id,
    uint64_t since_epoch, const uint8_t* cursor, size_t cursor_len,
    size_t max_events, size_t max_bytes);
/* Borrowed pointers remain valid until page free; never free them separately.
 * Event coordinates (including edge endpoints) are exact decimal strings.
 * Null page yields NULL pointers and zero count. */
const char* grafeo_change_page_events_json(const GrafeoChangePage* page);
const uint8_t* grafeo_change_page_cursor(const GrafeoChangePage* page);
size_t grafeo_change_page_event_count(const GrafeoChangePage* page);
void grafeo_free_change_page(GrafeoChangePage* page);


/* ---- Query execution ----------------------------------------------------- */

GrafeoResult* grafeo_execute(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_cypher(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_cypher_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_gremlin(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_gremlin_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_graphql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_graphql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_sparql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sparql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* Requires sql-pgq feature. */
GrafeoResult* grafeo_execute_sql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* Unified language dispatcher: language is "gql", "cypher", "gremlin", "graphql", "sparql", or "sql".
 * params_json may be NULL. */
GrafeoResult* grafeo_execute_language(GrafeoDatabase* db, const char* language, const char* query, const char* params_json);

/* ---- Result access ------------------------------------------------------- */

const char* grafeo_result_json(const GrafeoResult* result);
size_t      grafeo_result_row_count(const GrafeoResult* result);
double      grafeo_result_execution_time_ms(const GrafeoResult* result);
uint64_t    grafeo_result_rows_scanned(const GrafeoResult* result);
const char* grafeo_result_nodes_json(const GrafeoResult* result);
const char* grafeo_result_edges_json(const GrafeoResult* result);
void        grafeo_free_result(GrafeoResult* result);

/* ---- Bounded streaming -------------------------------------------------- */
/* Pull-only read cursors retain native query ownership. Unsupported blocking
 * operators and mutation routes return a structured error. Explicit max_rows
 * caps total delivered rows; max_bytes caps each copied row/chunk. NULL options
 * uses default per-copy bytes without imposing a total stream row limit.
 * Terminal errors remain observable on later pulls/close. */

GrafeoStream* grafeo_stream_open_with_options(GrafeoDatabase* db, const char* query,
    const char* params_json, const GrafeoQueryOptions* options);
/* Returns at most max_rows (internally capped at 1024); zero is invalid.
 * A non-NULL result is owned and freed by grafeo_free_result; NULL means EOF.
 * Row and chunk pulls advance the same cursor and may be interleaved. */
GrafeoStatus grafeo_stream_next_chunk(GrafeoStream* stream, size_t max_rows, GrafeoResult** out_result);
/* Idempotent fallible close; may interrupt a concurrent pull. Free the stream
 * only after all concurrent uses of that allocation have ended. */
GrafeoStatus grafeo_stream_close(GrafeoStream* stream);

/* Opens a streaming query. Returns NULL on error (check grafeo_last_error). */
GrafeoStream* grafeo_stream_open(GrafeoDatabase* db, const char* query);

/* Returns the column names as a JSON array. Caller frees with grafeo_free_string. */
char* grafeo_stream_columns_json(const GrafeoStream* stream);

/* Pulls the next row into *out_json as a JSON object.
 * On success (GRAFEO_OK) with non-NULL *out_json: caller frees the
 * string with grafeo_free_string. On success with NULL *out_json: stream
 * exhausted, stop iterating. On any other status, grafeo_last_error has
 * details. */
GrafeoStatus grafeo_stream_next_row_json(GrafeoStream* stream, char** out_json);

/* Frees a stream handle. Safe to call on NULL. */
void grafeo_stream_free(GrafeoStream* stream);

/* ---- Schema context ------------------------------------------------------ */

GrafeoStatus    grafeo_set_schema(GrafeoDatabase* db, const char* name);
GrafeoStatus    grafeo_reset_schema(GrafeoDatabase* db);
const char*     grafeo_current_schema(const GrafeoDatabase* db);

/* ---- Node CRUD ----------------------------------------------------------- */

uint64_t     grafeo_create_node(GrafeoDatabase* db, const char* labels_json, const char* properties_json);
GrafeoStatus grafeo_get_node(GrafeoDatabase* db, uint64_t id, GrafeoNode** out);
int32_t      grafeo_delete_node(GrafeoDatabase* db, uint64_t id);
GrafeoStatus grafeo_set_node_property(GrafeoDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      grafeo_remove_node_property(GrafeoDatabase* db, uint64_t id, const char* key);
int32_t      grafeo_add_node_label(GrafeoDatabase* db, uint64_t id, const char* label);
int32_t      grafeo_remove_node_label(GrafeoDatabase* db, uint64_t id, const char* label);
char*        grafeo_get_node_labels(GrafeoDatabase* db, uint64_t id);

uint64_t    grafeo_node_id(const GrafeoNode* node);
const char* grafeo_node_labels_json(const GrafeoNode* node);
const char* grafeo_node_properties_json(const GrafeoNode* node);
void        grafeo_free_node(GrafeoNode* node);

/* ---- Edge CRUD ----------------------------------------------------------- */

uint64_t     grafeo_create_edge(GrafeoDatabase* db, uint64_t source_id, uint64_t target_id, const char* edge_type, const char* properties_json);
GrafeoStatus grafeo_get_edge(GrafeoDatabase* db, uint64_t id, GrafeoEdge** out);
int32_t      grafeo_delete_edge(GrafeoDatabase* db, uint64_t id);
GrafeoStatus grafeo_set_edge_property(GrafeoDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      grafeo_remove_edge_property(GrafeoDatabase* db, uint64_t id, const char* key);

uint64_t    grafeo_edge_id(const GrafeoEdge* edge);
uint64_t    grafeo_edge_source_id(const GrafeoEdge* edge);
uint64_t    grafeo_edge_target_id(const GrafeoEdge* edge);
const char* grafeo_edge_type(const GrafeoEdge* edge);
const char* grafeo_edge_properties_json(const GrafeoEdge* edge);
void        grafeo_free_edge(GrafeoEdge* edge);

/* ---- Catalog-owned index mutations --------------------------------------- */
/* UTF-8 spans are pointer+byte-length, not NUL-terminated. NULL is legal only
 * for zero length. Graph count 0 selects root; [empty span] is an empty-named
 * graph component and is distinct from root. All memory is caller-owned and
 * valid/aligned through the call. Outputs are written only on success. */
typedef struct {
    const uint8_t* data;
    size_t len;
} GrafeoUtf8;

#define GRAFEO_INDEX_PROPERTY ((uint32_t)0)
#define GRAFEO_INDEX_BTREE    ((uint32_t)1)
#define GRAFEO_INDEX_TEXT     ((uint32_t)2)
#define GRAFEO_INDEX_VECTOR   ((uint32_t)3)
#define GRAFEO_INDEX_NAME_PRESENT         ((uint32_t)1)
#define GRAFEO_INDEX_LABEL_PRESENT        ((uint32_t)2)
#define GRAFEO_INDEX_DIMENSIONS_PRESENT   ((uint32_t)4)
#define GRAFEO_INDEX_METRIC_PRESENT       ((uint32_t)8)
#define GRAFEO_INDEX_M_PRESENT            ((uint32_t)16)
#define GRAFEO_INDEX_EF_CONSTRUCTION_PRESENT ((uint32_t)32)
#define GRAFEO_INDEX_QUANTIZATION_PRESENT ((uint32_t)64)
#define GRAFEO_INDEX_MIN_TOKEN_LENGTH_PRESENT ((uint32_t)128)

typedef struct {
    uint32_t kind;
    uint32_t options;
    const GrafeoUtf8* graph;
    size_t graph_count;
    GrafeoUtf8 name;
    GrafeoUtf8 label;
    GrafeoUtf8 property;
    GrafeoUtf8 metric;
    GrafeoUtf8 quantization;
    size_t dimensions;
    size_t m;
    size_t ef_construction;
    /* Text only; presence bit 128. Omission uses the default; zero is valid. */
    size_t min_token_length;
} GrafeoIndexRequest;

GrafeoStatus grafeo_create_index(GrafeoDatabase* db, const GrafeoIndexRequest* request, uint32_t* out_id);
GrafeoStatus grafeo_drop_index(GrafeoDatabase* db, uint32_t id, int32_t* out_dropped);
GrafeoStatus grafeo_rebuild_index(GrafeoDatabase* db, uint32_t id);

/* ---- Property index reads ------------------------------------------------ */
int32_t      grafeo_has_property_index(GrafeoDatabase* db, const char* property);
GrafeoStatus grafeo_find_nodes_by_property(GrafeoDatabase* db, const char* property, const char* value_json, uint64_t** out_ids, size_t* out_count);
void         grafeo_free_node_ids(uint64_t* ids, size_t count);

/* ---- Vector operations --------------------------------------------------- */


GrafeoStatus grafeo_vector_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
GrafeoStatus grafeo_mmr_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t fetch_k, float lambda, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
/* Requires vector-index feature. *out_ids has *out_count entries; free with grafeo_free_node_ids(*out_ids, *out_count). */
GrafeoStatus grafeo_batch_create_nodes(GrafeoDatabase* db, const char* label, const char* property, const float* vectors, size_t vector_count, size_t dimensions, uint64_t** out_ids, size_t* out_count);
void         grafeo_free_vector_results(uint64_t* ids, float* distances, size_t count);

/* ---- Statistics ---------------------------------------------------------- */

size_t grafeo_node_count(GrafeoDatabase* db);
size_t grafeo_edge_count(GrafeoDatabase* db);

/* ---- Transactions -------------------------------------------------------- */

GrafeoTransaction* grafeo_begin_transaction(GrafeoDatabase* db);
GrafeoTransaction* grafeo_begin_transaction_with_isolation(GrafeoDatabase* db, GrafeoIsolationLevel isolation);
GrafeoResult*      grafeo_transaction_execute(GrafeoTransaction* tx, const char* query);
GrafeoResult*      grafeo_transaction_execute_with_params(GrafeoTransaction* tx, const char* query, const char* params_json);
/* Execute in a specific language within a transaction; params_json may be NULL. */
GrafeoResult*      grafeo_transaction_execute_language(GrafeoTransaction* tx, const char* language, const char* query, const char* params_json);
GrafeoStatus       grafeo_commit(GrafeoTransaction* tx);
/* Same as grafeo_commit; writes the assigned epoch when out_epoch is non-NULL. */
GrafeoStatus       grafeo_commit_epoch(GrafeoTransaction* tx, uint64_t* out_epoch);
GrafeoStatus       grafeo_rollback(GrafeoTransaction* tx);
void               grafeo_free_transaction(GrafeoTransaction* tx);
uint64_t           grafeo_transaction_create_node(GrafeoTransaction* tx, const char* labels_json, const char* properties_json);

/* ---- RDF quads (N-Triples term strings; graph NULL = default graph) ------ */
/* Requires the triple-store / native / rdf compile set. Without it the
 * symbols exist but return GRAFEO_ERROR_DATABASE. */

GrafeoStatus grafeo_insert_rdf_quad(GrafeoDatabase* db, const char* subject, const char* predicate, const char* object, const char* graph);
/* Bulk insert. graphs may be NULL (all default). graphs[i] may be NULL. out_* may be NULL. */
GrafeoStatus grafeo_insert_rdf_quads(GrafeoDatabase* db, const char* const* subjects, const char* const* predicates, const char* const* objects, const char* const* graphs, size_t count, size_t* out_inserted, uint64_t* out_epoch);
bool         grafeo_contains_rdf_quad(GrafeoDatabase* db, const char* subject, const char* predicate, const char* object, const char* graph);
GrafeoStatus grafeo_transaction_insert_rdf_quad(GrafeoTransaction* tx, const char* subject, const char* predicate, const char* object, const char* graph);
GrafeoStatus grafeo_transaction_insert_rdf_quads(GrafeoTransaction* tx, const char* const* subjects, const char* const* predicates, const char* const* objects, const char* const* graphs, size_t count, size_t* out_inserted);
bool         grafeo_transaction_contains_rdf_quad(GrafeoTransaction* tx, const char* subject, const char* predicate, const char* object, const char* graph);

/* ---- Admin --------------------------------------------------------------- */

char*        grafeo_info(GrafeoDatabase* db);
GrafeoStatus grafeo_save(GrafeoDatabase* db, const char* path);
GrafeoStatus grafeo_wal_checkpoint(GrafeoDatabase* db);
/* Repeatable LPG compaction; retained history and subsequent writes remain supported.
 * Requires no active transactions or live Sessions. Check status on failure.
 * This is not a durability checkpoint or a history-retention lease. */
GrafeoStatus grafeo_compact(GrafeoDatabase* db);

/* ---- Memory management --------------------------------------------------- */

void grafeo_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* GRAFEO_H */
