/* Compile against the candidate header and link the exact profile library. */
#include "../grafeo.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(GRAFEO_EXPECT_GQL) || !defined(GRAFEO_EXPECT_STORAGE)
#error "Set the compiled profile's query and storage expectations"
#endif

static void node_value(GrafeoDatabase* db, uint64_t id, const char* expected) {
    GrafeoNode* node = NULL;
    assert(grafeo_get_node(db, id, &node) == GRAFEO_OK && node);
    assert(grafeo_node_id(node) == id);
    assert(strcmp(grafeo_node_properties_json(node), expected) == 0);
    grafeo_free_node(node);
}

static void edge_value(GrafeoDatabase* db, uint64_t id, uint64_t source,
                       uint64_t target, const char* expected) {
    GrafeoEdge* edge = NULL;
    assert(grafeo_get_edge(db, id, &edge) == GRAFEO_OK && edge);
    assert(grafeo_edge_id(edge) == id);
    assert(grafeo_edge_source_id(edge) == source);
    assert(grafeo_edge_target_id(edge) == target);
    assert(strcmp(grafeo_edge_type(edge), "LINK") == 0);
    assert(strcmp(grafeo_edge_properties_json(edge), expected) == 0);
    grafeo_free_edge(edge);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    GrafeoDatabase* db = grafeo_open_memory();
    assert(db);
    uint64_t first = grafeo_create_node(db, "[\"Node\"]", "{\"value\":42}");
    uint64_t second = grafeo_create_node(db, "[\"Node\"]", "{\"value\":99}");
    assert(first != UINT64_MAX && second != UINT64_MAX && first != second);
    uint64_t left = grafeo_create_edge(db, first, second, "LINK", "{\"weight\":7}");
    uint64_t right = grafeo_create_edge(db, first, second, "LINK", "{\"weight\":8}");
    assert(left != UINT64_MAX && right != UINT64_MAX && left != right);
    node_value(db, first, "{\"value\":42}");
    edge_value(db, left, first, second, "{\"weight\":7}");
    edge_value(db, right, first, second, "{\"weight\":8}");
    assert(grafeo_node_count(db) == 2 && grafeo_edge_count(db) == 2);
    puts("PASS C profile: direct CRUD and parallel edges");

    GrafeoTransaction* tx = grafeo_begin_transaction_with_isolation(db, GRAFEO_ISOLATION_SERIALIZABLE);
    assert(tx);
    uint64_t kept = grafeo_transaction_create_node(tx, "[\"Committed\"]", "{\"value\":17}");
    assert(kept != UINT64_MAX);
    GrafeoNode* missing = NULL;
    assert(grafeo_get_node(db, kept, &missing) == GRAFEO_ERROR_DATABASE && !missing);
    assert(grafeo_commit(tx) == GRAFEO_OK);
    grafeo_free_transaction(tx);
    node_value(db, kept, "{\"value\":17}");
    tx = grafeo_begin_transaction(db);
    assert(tx);
    uint64_t removed = grafeo_transaction_create_node(tx, "[\"RolledBack\"]", NULL);
    assert(removed != UINT64_MAX);
    assert(grafeo_rollback(tx) == GRAFEO_OK);
    grafeo_free_transaction(tx);
    assert(grafeo_get_node(db, removed, &missing) == GRAFEO_ERROR_DATABASE && !missing);
    assert(grafeo_node_count(db) == 3);
    puts("PASS C profile: transaction isolation and commit/rollback visibility");

    GrafeoIndexRequest request = {0};
    request.kind = GRAFEO_INDEX_PROPERTY;
    request.property = (GrafeoUtf8){(const uint8_t*)"value", 5};
    uint32_t owner = UINT32_MAX;
    assert(grafeo_create_index(db, &request, &owner) == GRAFEO_OK);
    assert(owner != UINT32_MAX && grafeo_has_property_index(db, "value") == 1);
    uint64_t* ids = NULL;
    size_t count = 0;
    assert(grafeo_find_nodes_by_property(db, "value", "42", &ids, &count) == GRAFEO_OK);
    assert(count == 1 && ids[0] == first);
    grafeo_free_node_ids(ids, count);
    puts("PASS C profile: property index exports and exact lookup");

    assert(grafeo_compact(db) == GRAFEO_OK);
    edge_value(db, left, first, second, "{\"weight\":7}");
    edge_value(db, right, first, second, "{\"weight\":8}");
    assert(grafeo_edge_count(db) == 2);
    assert(grafeo_set_node_property(db, first, "value", "43") == GRAFEO_OK);
    assert(grafeo_delete_edge(db, left) == 1);
    for (unsigned round = 0; round < 2; ++round) {
        node_value(db, first, "{\"value\":43}");
        edge_value(db, right, first, second, "{\"weight\":8}");
        assert(grafeo_node_count(db) == 3 && grafeo_edge_count(db) == 1);
        assert(grafeo_compact(db) == GRAFEO_OK);
    }
    assert(grafeo_rebuild_index(db, owner) == GRAFEO_OK);
    ids = NULL;
    count = 0;
    assert(grafeo_find_nodes_by_property(db, "value", "43", &ids, &count) == GRAFEO_OK);
    assert(count == 1 && ids[0] == first);
    grafeo_free_node_ids(ids, count);
    int32_t dropped = 0;
    assert(grafeo_drop_index(db, owner, &dropped) == GRAFEO_OK && dropped == 1);
    assert(grafeo_has_property_index(db, "value") == 0);
    puts("PASS C profile: compact/recompact, overlay deletion and index lifecycle");

    const char* subjects[] = {"<urn:s>"};
    const char* predicates[] = {"<urn:p>"};
    const char* objects[] = {"\"value\""};
    size_t inserted = 99;
    uint64_t epoch = 99;
    assert(grafeo_insert_rdf_quad(db, subjects[0], predicates[0], objects[0], NULL) == GRAFEO_ERROR_DATABASE);
    assert(grafeo_insert_rdf_quads(db, subjects, predicates, objects, NULL, 1, &inserted, &epoch) == GRAFEO_ERROR_DATABASE);
    assert(inserted == 99 && epoch == 99);
    assert(!grafeo_contains_rdf_quad(db, subjects[0], predicates[0], objects[0], NULL));
    tx = grafeo_begin_transaction(db);
    assert(tx);
    assert(grafeo_transaction_insert_rdf_quad(tx, subjects[0], predicates[0], objects[0], NULL) == GRAFEO_ERROR_DATABASE);
    assert(grafeo_transaction_insert_rdf_quads(tx, subjects, predicates, objects, NULL, 1, &inserted) == GRAFEO_ERROR_DATABASE);
    assert(inserted == 99);
    assert(!grafeo_transaction_contains_rdf_quad(tx, subjects[0], predicates[0], objects[0], NULL));
    assert(grafeo_rollback(tx) == GRAFEO_OK);
    grafeo_free_transaction(tx);
    assert(grafeo_node_count(db) == 3 && grafeo_edge_count(db) == 1);
    puts("PASS C profile: six unavailable RDF calls fail without mutation");

    GrafeoResult* result = grafeo_execute(db, "RETURN 1 AS value");
#if GRAFEO_EXPECT_GQL
    assert(result && grafeo_result_row_count(result) == 1);
    assert(strcmp(grafeo_result_json(result), "[{\"value\":1}]") == 0);
    grafeo_free_result(result);
#else
    assert(!result);
#endif
    puts("PASS C profile: parser availability");

#if GRAFEO_EXPECT_STORAGE
    assert(grafeo_save(db, argv[1]) == GRAFEO_OK);
    GrafeoDatabase* reopened = grafeo_open(argv[1]);
    assert(reopened);
    node_value(reopened, first, "{\"value\":43}");
    edge_value(reopened, right, first, second, "{\"weight\":8}");
    assert(grafeo_close(reopened) == GRAFEO_OK);
    grafeo_free_database(reopened);
    assert(remove(argv[1]) == 0);
#else
    assert(grafeo_save(db, argv[1]) == GRAFEO_ERROR_STORAGE);
#endif
    assert(grafeo_close(db) == GRAFEO_OK);
    grafeo_free_database(db);
    puts("PASS C profile: storage availability and cleanup");
    return 0;
}
