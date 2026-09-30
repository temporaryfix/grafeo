/* Link against each exact RDF-capable C profile; no LPG symbols required. */
#include "../grafeo.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(GRAFEO_EXPECT_SPARQL) || !defined(GRAFEO_EXPECT_STORAGE)
#error "Set the compiled profile's query and storage expectations"
#endif

static void metadata(GrafeoDatabase* db, int persistent) {
    assert(grafeo_graph_model(db) == GRAFEO_GRAPH_MODEL_RDF);
    char* info = grafeo_info(db);
    assert(info);
    assert(strstr(info, persistent ? "\"is_persistent\":true" : "\"is_persistent\":false"));
    if (!persistent) assert(strstr(info, "\"path\":null"));
    grafeo_free_string(info);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    GrafeoDatabase* db = grafeo_open_memory_model(GRAFEO_GRAPH_MODEL_RDF);
    assert(db);
    metadata(db, 0);
    puts("PASS C RDF: shared metadata");

    const char* subjects[] = {"<urn:s>", "<urn:s>"};
    const char* predicates[] = {"<urn:p>", "<urn:p>"};
    const char* objects[] = {"\"雪\"", "\"雪\""};
    const char* graphs[] = {"urn:g", "urn:g"};
    size_t inserted = 99;
    uint64_t epoch = 0;
    assert(grafeo_insert_rdf_quads(db, subjects, predicates, objects, graphs, 2, &inserted, &epoch) == GRAFEO_OK);
    assert(inserted == 1 && epoch > 0);
    assert(grafeo_contains_rdf_quad(db, subjects[0], predicates[0], objects[0], graphs[0]));
    assert(!grafeo_contains_rdf_quad(db, subjects[0], predicates[0], objects[0], NULL));
    assert(grafeo_insert_rdf_quads(db, subjects, predicates, objects, graphs, 2, &inserted, &epoch) == GRAFEO_OK);
    assert(inserted == 0);
    const char* typed = "\"007\"^^<http://www.w3.org/2001/XMLSchema#integer>";
    assert(grafeo_insert_rdf_quad(db, "<urn:number>", "<urn:p>", typed, NULL) == GRAFEO_OK);
    assert(grafeo_contains_rdf_quad(db, "<urn:number>", "<urn:p>", typed, NULL));
    assert(!grafeo_contains_rdf_quad(db, "<urn:number>", "<urn:p>", "\"7\"^^<http://www.w3.org/2001/XMLSchema#integer>", NULL));
    puts("PASS C RDF: exact quad membership, graph, datatype and duplicate receipt");

    GrafeoTransaction* tx = grafeo_begin_transaction_with_isolation(db, GRAFEO_ISOLATION_SERIALIZABLE);
    assert(tx);
    assert(grafeo_transaction_insert_rdf_quad(tx, "<urn:kept>", "<urn:p>", "\"kept\"", NULL) == GRAFEO_OK);
    assert(grafeo_transaction_contains_rdf_quad(tx, "<urn:kept>", "<urn:p>", "\"kept\"", NULL));
    assert(!grafeo_contains_rdf_quad(db, "<urn:kept>", "<urn:p>", "\"kept\"", NULL));
    uint64_t committed = 0;
    assert(grafeo_commit_epoch(tx, &committed) == GRAFEO_OK && committed > 0);
    grafeo_free_transaction(tx);
    assert(grafeo_contains_rdf_quad(db, "<urn:kept>", "<urn:p>", "\"kept\"", NULL));
    tx = grafeo_begin_transaction(db);
    assert(tx);
    assert(grafeo_transaction_insert_rdf_quad(tx, "<urn:removed>", "<urn:p>", "\"removed\"", NULL) == GRAFEO_OK);
    assert(grafeo_rollback(tx) == GRAFEO_OK);
    grafeo_free_transaction(tx);
    assert(!grafeo_contains_rdf_quad(db, "<urn:removed>", "<urn:p>", "\"removed\"", NULL));
    puts("PASS C RDF: explicit isolation, commit epoch and rollback visibility");

    const char* bad_subjects[] = {"<urn:must-not-appear>", "\"invalid subject\""};
    inserted = 99;
    epoch = 99;
    assert(grafeo_insert_rdf_quads(db, bad_subjects, predicates, objects, graphs, 2, &inserted, &epoch) != GRAFEO_OK);
    assert(inserted == 99 && epoch == 99);
    assert(!grafeo_contains_rdf_quad(db, bad_subjects[0], predicates[0], objects[0], graphs[0]));
    assert(grafeo_contains_rdf_quad(db, subjects[0], predicates[0], objects[0], graphs[0]));
    puts("PASS C RDF: malformed bulk input rejects before mutation or output writes");

    GrafeoQueryOptions options = {NULL, 10, 65536, "sparql"};
    GrafeoResult* result = grafeo_execute_with_options(db,
        "SELECT ?s WHERE { GRAPH <urn:g> { ?s <urn:p> \"雪\" } }", NULL, &options);
#if GRAFEO_EXPECT_SPARQL
    assert(result && grafeo_result_row_count(result) == 1);
    assert(strstr(grafeo_result_json(result), "urn:s"));
    grafeo_free_result(result);
#else
    assert(!result);
    assert(strcmp(grafeo_last_error_code(), "GRAFEO-Q002") == 0);
#endif
    puts("PASS C RDF: query availability and bounded result ownership");

#if GRAFEO_EXPECT_STORAGE
    assert(grafeo_save(db, argv[1]) == GRAFEO_OK);
    GrafeoDatabase* reopened = grafeo_open(argv[1]);
    assert(reopened);
    metadata(reopened, 1);
    assert(grafeo_contains_rdf_quad(reopened, subjects[0], predicates[0], objects[0], graphs[0]));
    assert(grafeo_contains_rdf_quad(reopened, "<urn:number>", "<urn:p>", typed, NULL));
    assert(grafeo_contains_rdf_quad(reopened, "<urn:kept>", "<urn:p>", "\"kept\"", NULL));
    assert(!grafeo_contains_rdf_quad(reopened, "<urn:removed>", "<urn:p>", "\"removed\"", NULL));
    assert(grafeo_close(reopened) == GRAFEO_OK);
    grafeo_free_database(reopened);
    assert(remove(argv[1]) == 0);
#else
    assert(grafeo_save(db, argv[1]) == GRAFEO_ERROR_STORAGE);
#endif
#if defined(GRAFEO_CHECK_STREAM_UNAVAILABLE)
    assert(!grafeo_stream_open(db, "RETURN 1"));
    assert(!grafeo_stream_open_with_options(db, "RETURN 1", NULL, &options));
    assert(strcmp(grafeo_last_error_code(), "GRAFEO-Q004") == 0);
    assert(!grafeo_stream_columns_json(NULL));
    char* row = (char*)(uintptr_t)1;
    assert(grafeo_stream_next_row_json(NULL, &row) == GRAFEO_ERROR_QUERY && !row);
    GrafeoResult* chunk = (GrafeoResult*)(uintptr_t)1;
    assert(grafeo_stream_next_chunk(NULL, 1, &chunk) == GRAFEO_ERROR_QUERY && !chunk);
    assert(strcmp(grafeo_last_error_code(), "GRAFEO-Q004") == 0);
    assert(grafeo_stream_close(NULL) == GRAFEO_OK);
    grafeo_stream_free(NULL);
    assert(grafeo_contains_rdf_quad(db, "<urn:kept>", "<urn:p>", "\"kept\"", NULL));
    puts("PASS C RDF: unavailable LPG stream exports reject and clear outputs");
#endif
    assert(grafeo_close(db) == GRAFEO_OK);
    grafeo_free_database(db);
    puts("PASS C RDF: save/reopen capability and exact retained state");
    return 0;
}
