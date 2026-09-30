/* Compile and run against the exact candidate header/shared library. */
#include "../grafeo.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(offsetof(GrafeoQueryOptions, control) == 0, "control ABI");
_Static_assert(offsetof(GrafeoQueryOptions, max_rows) == sizeof(void*), "rows ABI");
_Static_assert(offsetof(GrafeoQueryOptions, max_bytes) == sizeof(void*) + sizeof(size_t), "bytes ABI");
_Static_assert(offsetof(GrafeoQueryOptions, language) == sizeof(void*) + 2*sizeof(size_t), "language ABI");
_Static_assert(GRAFEO_ERROR_CANCELLED == 10 && GRAFEO_ERROR_DEADLINE == 11 && GRAFEO_ERROR_RESOURCE_LIMIT == 12, "status ABI");

int main(void) {
    GrafeoDatabase* db = grafeo_open_memory();
    assert(db);
    GrafeoQueryControl* control = grafeo_query_control_create(-1);
    assert(control);
    GrafeoCancelHandle* cancel = grafeo_query_control_cancel_handle(control);
    GrafeoCancelHandle* clone = grafeo_cancel_handle_clone(cancel);
    assert(cancel && clone);
    grafeo_cancel_handle_free(cancel);
    assert(grafeo_cancel(clone) == GRAFEO_OK);
    GrafeoQueryOptions options = {control, 100, 65536, "gql"};
    assert(!grafeo_execute_with_options(db, "CREATE (:Denied)", NULL, &options));
    assert(strcmp(grafeo_last_error_code(), "GRAFEO-Q007") == 0);
    grafeo_query_control_free(control);
    grafeo_cancel_handle_free(clone);

    control = grafeo_query_control_create(0);
    assert(control);
    options.control = control;
    assert(!grafeo_execute_with_options(db, "RETURN 1", NULL, &options));
    assert(strcmp(grafeo_last_error_code(), "GRAFEO-Q003") == 0);
    grafeo_query_control_free(control);

    options.control = NULL;
    options.max_bytes = 0;
    assert(!grafeo_execute_with_options(db, "CREATE (:Denied) RETURN 1", NULL, &options));
    assert(strcmp(grafeo_last_error_code(), "GRAFEO-S001") == 0);
    GrafeoResult* result = grafeo_execute(db, "MATCH (n:Denied) RETURN n");
    assert(result && grafeo_result_row_count(result) == 0);
    grafeo_free_result(result);

    options.max_bytes = 65536;
    GrafeoStream* stream = grafeo_stream_open_with_options(db, "UNWIND [1,2,3] AS value RETURN value", NULL, &options);
    assert(stream);
    assert(grafeo_close(db) != GRAFEO_OK);
    result = NULL;
    assert(grafeo_stream_next_chunk(stream, 2, &result) == GRAFEO_OK);
    assert(result && grafeo_result_row_count(result) == 2);
    grafeo_free_result(result);
    char* row = NULL;
    assert(grafeo_stream_next_row_json(stream, &row) == GRAFEO_OK);
    assert(row && strstr(row, "3"));
    grafeo_free_string(row);
    assert(grafeo_stream_close(stream) == GRAFEO_OK);
    assert(grafeo_stream_close(stream) == GRAFEO_OK);
    grafeo_stream_free(stream);
    assert(grafeo_close(db) == GRAFEO_OK);
    grafeo_free_database(db);
    puts("PASS C ABI layout/control/deadline/copy-rollback/chunk/close: 6 groups");
    return 0;
}
