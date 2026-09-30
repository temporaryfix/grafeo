// Package grafeo provides Go bindings for the Grafeo graph database.
//
// It uses CGO to link against the grafeo-c shared library, which provides
// a C-compatible FFI layer on top of the Rust engine.
//
// Quick start:
//
//	db, err := grafeo.OpenInMemory()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer db.Close()
//
//	db.Execute(`CREATE (:Person {name: 'Alix', age: 30})`)
//	result, _ := db.Execute(`MATCH (p:Person) RETURN p.name`)
package grafeo

/*
#cgo LDFLAGS: -lgrafeo_c
#cgo linux LDFLAGS: -lm -ldl -lpthread
#cgo darwin LDFLAGS: -lm -ldl -lpthread -framework Security
#cgo windows LDFLAGS: -lws2_32 -lbcrypt -lntdll -luserenv

#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"
)

// Database is the primary handle to a Grafeo graph database.
// It is safe for concurrent use from multiple goroutines.
type Database struct {
	mu     sync.RWMutex
	handle *C.GrafeoDatabase
}

// OpenInMemory creates a new in-memory database.
func OpenInMemory() (*Database, error) {
	runtime.LockOSThread()
	h := C.grafeo_open_memory()
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// Open opens or creates a persistent database at the given path.
func Open(path string) (*Database, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	runtime.LockOSThread()
	h := C.grafeo_open(cPath)
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// OpenSingleFile opens or creates a persistent database in single-file
// `.grafeo` format at the given path, bypassing the Auto storage-format
// detection based on path extension.
func OpenSingleFile(path string) (*Database, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	runtime.LockOSThread()
	h := C.grafeo_open_single_file(cPath)
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	db := &Database{handle: h}
	runtime.SetFinalizer(db, (*Database).free)
	return db, nil
}

// Close flushes any pending writes and releases the database handle.
func (db *Database) Close() error {
	if !db.mu.TryLock() {
		return ErrBusy
	}
	defer db.mu.Unlock()
	defer runtime.KeepAlive(db)
	if db.handle == nil {
		return nil
	}
	runtime.LockOSThread()
	err := statusToError(C.grafeo_close(db.handle))
	runtime.UnlockOSThread()
	if err != nil {
		return err
	}
	C.grafeo_free_database(db.handle)
	db.handle = nil
	runtime.SetFinalizer(db, nil)
	return nil
}

// acquire retains the database handle throughout a native call. Contention with
// Close is reported immediately, so context callers never wait on an owner lock.
func (db *Database) acquire() error {
	if !db.mu.TryRLock() {
		return ErrBusy
	}
	if db.handle == nil {
		db.mu.RUnlock()
		return ErrClosed
	}
	return nil
}
func (db *Database) release() { db.mu.RUnlock(); runtime.KeepAlive(db) }

// free is called only when no Go owner can still access the handle.
func (db *Database) free() { _ = db.Close() }

// ExecuteContext runs a query with context cancellation and execution limits.
func (db *Database) ExecuteContext(ctx context.Context, query string, params map[string]any, options *ExecutionOptions) (*QueryResult, error) {
	paramsJSON, err := marshalExecutionParams(params)
	if err != nil {
		return nil, err
	}
	return db.executeContextJSON(ctx, query, paramsJSON, options)
}

func marshalExecutionParams(params map[string]any) (string, error) {
	if params == nil {
		return "", nil
	}
	data, err := json.Marshal(params)
	if err != nil {
		return "", fmt.Errorf("%w: failed to marshal params: %v", ErrDatabase, err)
	}
	return string(data), nil
}

func validateExecutionStrings(query, paramsJSON string) error {
	if strings.IndexByte(query, 0) >= 0 || strings.IndexByte(paramsJSON, 0) >= 0 || !utf8.ValidString(query) || !utf8.ValidString(paramsJSON) {
		return invalidExecution("query and parameters must be valid UTF-8 without NUL")
	}
	if paramsJSON != "" {
		var object map[string]json.RawMessage
		if err := json.Unmarshal([]byte(paramsJSON), &object); err != nil || object == nil {
			return invalidExecution("parameters must be a JSON object")
		}
	}
	return nil
}

func (db *Database) executeContextJSON(ctx context.Context, query, paramsJSON string, options *ExecutionOptions) (*QueryResult, error) {
	if err := validateExecutionStrings(query, paramsJSON); err != nil {
		return nil, err
	}
	if err := db.acquire(); err != nil {
		return nil, err
	}
	defer db.release()
	inv, err := newInvocation(ctx, options, false)
	if err != nil {
		return nil, err
	}
	defer inv.finish()
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	var cParams *C.char
	if paramsJSON != "" {
		cParams = C.CString(paramsJSON)
		defer C.free(unsafe.Pointer(cParams))
	}
	runtime.LockOSThread()
	r := C.grafeo_execute_with_options(db.handle, cQuery, cParams, &inv.options)
	if r == nil {
		err = lastError()
	}
	runtime.UnlockOSThread()
	if err != nil {
		return nil, inv.error(err)
	}
	defer C.grafeo_free_result(r)
	return parseResultBounded(r, inv.maxBytes)
}

// Execute runs a GQL query and returns the results.
func (db *Database) Execute(query string) (*QueryResult, error) {
	return db.executeContextJSON(context.Background(), query, "", nil)
}

// ExecuteParams runs a GQL query with parameters as a Go map.
func (db *Database) ExecuteParams(query string, params map[string]any) (*QueryResult, error) {
	return db.ExecuteContext(context.Background(), query, params, nil)
}

// ExecuteWithParams runs a GQL query with JSON object parameters.
func (db *Database) ExecuteWithParams(query, paramsJSON string) (*QueryResult, error) {
	return db.executeContextJSON(context.Background(), query, paramsJSON, nil)
}

// ExecuteLanguage runs a query in the given language with optional JSON parameters.
func (db *Database) ExecuteLanguage(language, query, paramsJSON string) (*QueryResult, error) {
	return db.executeContextJSON(context.Background(), query, paramsJSON, &ExecutionOptions{Language: language})
}

// ExecuteCypher runs a Cypher query.
func (db *Database) ExecuteCypher(query string) (*QueryResult, error) {
	return db.ExecuteLanguage("cypher", query, "")
}

// ExecuteCypherWithParams runs a Cypher query with JSON object parameters.
func (db *Database) ExecuteCypherWithParams(query, paramsJSON string) (*QueryResult, error) {
	return db.ExecuteLanguage("cypher", query, paramsJSON)
}

// ExecuteGremlin runs a Gremlin query.
func (db *Database) ExecuteGremlin(query string) (*QueryResult, error) {
	return db.ExecuteLanguage("gremlin", query, "")
}

// ExecuteGremlinWithParams runs a Gremlin query with JSON object parameters.
func (db *Database) ExecuteGremlinWithParams(query, paramsJSON string) (*QueryResult, error) {
	return db.ExecuteLanguage("gremlin", query, paramsJSON)
}

// ExecuteGraphQL runs a GraphQL query.
func (db *Database) ExecuteGraphQL(query string) (*QueryResult, error) {
	return db.ExecuteLanguage("graphql", query, "")
}

// ExecuteGraphQLWithParams runs a GraphQL query with JSON object parameters.
func (db *Database) ExecuteGraphQLWithParams(query, paramsJSON string) (*QueryResult, error) {
	return db.ExecuteLanguage("graphql", query, paramsJSON)
}

// ExecuteSPARQL runs a SPARQL query.
func (db *Database) ExecuteSPARQL(query string) (*QueryResult, error) {
	return db.ExecuteLanguage("sparql", query, "")
}

// ExecuteSPARQLWithParams runs a SPARQL query with JSON object parameters.
func (db *Database) ExecuteSPARQLWithParams(query, paramsJSON string) (*QueryResult, error) {
	return db.ExecuteLanguage("sparql", query, paramsJSON)
}

// ExecuteSQL runs a SQL query.
func (db *Database) ExecuteSQL(query string) (*QueryResult, error) {
	return db.ExecuteLanguage("sql", query, "")
}

// ExecuteSQLWithParams runs a SQL query with JSON object parameters.
func (db *Database) ExecuteSQLWithParams(query, paramsJSON string) (*QueryResult, error) {
	return db.ExecuteLanguage("sql", query, paramsJSON)
}

// MmrSearch finds diverse nearest neighbors using Maximal Marginal Relevance.
// fetchK is the number of HNSW candidates (use -1 for default 4*k).
// lambda controls relevance vs diversity (0=diverse, 1=relevant; use -1 for default 0.5).
// ef is the HNSW beam width (use -1 for default).
func (db *Database) MmrSearch(label, property string, query []float32, k int, fetchK int, lambda float32, ef int) ([]VectorResult, error) {
	if err := db.acquire(); err != nil {
		return nil, err
	}
	defer db.release()
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	var outIDs *C.uint64_t
	var outDists *C.float
	var outCount C.size_t

	runtime.LockOSThread()
	status := C.grafeo_mmr_search(
		db.handle, cLabel, cProp,
		(*C.float)(unsafe.Pointer(&query[0])), C.size_t(len(query)),
		C.size_t(k), C.int32_t(fetchK), C.float(lambda), C.int32_t(ef),
		&outIDs, &outDists, &outCount,
	)
	if status != C.GRAFEO_OK {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	count := int(outCount)
	if count == 0 {
		return nil, nil
	}
	defer C.grafeo_free_vector_results(outIDs, outDists, outCount)

	results := make([]VectorResult, count)
	ids := unsafe.Slice((*uint64)(unsafe.Pointer(outIDs)), count)
	dists := unsafe.Slice((*float32)(unsafe.Pointer(outDists)), count)
	for i := range count {
		results[i] = VectorResult{NodeID: ids[i], Distance: dists[i]}
	}
	return results, nil
}

// NodeCount returns the number of nodes in the database.
func (db *Database) NodeCount() int {
	if err := db.acquire(); err != nil {
		return 0
	}
	defer db.release()
	return int(C.grafeo_node_count(db.handle))
}

// EdgeCount returns the number of edges in the database.
func (db *Database) EdgeCount() int {
	if err := db.acquire(); err != nil {
		return 0
	}
	defer db.release()
	return int(C.grafeo_edge_count(db.handle))
}

// Version returns the Grafeo library version.
func Version() string {
	return C.GoString(C.grafeo_version())
}
