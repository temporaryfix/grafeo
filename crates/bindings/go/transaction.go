package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"runtime"
	"sync"
	"unsafe"
)

// Transaction represents an explicit transaction. Native operations are serialized;
// concurrent use returns ErrBusy. Unfinished transactions are rolled back on GC.
type Transaction struct {
	mu     sync.Mutex
	handle *C.GrafeoTransaction
	db     *Database
}

// BeginTransaction starts a transaction with snapshot isolation.
func (db *Database) BeginTransaction() (*Transaction, error) {
	return db.BeginTransactionWith(IsolationLevel(C.GRAFEO_ISOLATION_SNAPSHOT))
}

// BeginTransactionWith starts a transaction with a specific isolation level.
func (db *Database) BeginTransactionWith(level IsolationLevel) (*Transaction, error) {
	if err := db.acquire(); err != nil {
		return nil, err
	}
	defer db.release()
	runtime.LockOSThread()
	h := C.grafeo_begin_transaction_with_isolation(db.handle, C.GrafeoIsolationLevel(level))
	if h == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	tx := &Transaction{handle: h, db: db}
	runtime.SetFinalizer(tx, (*Transaction).free)
	return tx, nil
}

func (tx *Transaction) acquire() error {
	if !tx.mu.TryLock() {
		return ErrBusy
	}
	if tx.handle == nil {
		tx.mu.Unlock()
		return ErrClosed
	}
	if err := tx.db.acquire(); err != nil {
		tx.mu.Unlock()
		return err
	}
	return nil
}
func (tx *Transaction) release() {
	tx.db.release()
	tx.mu.Unlock()
	runtime.KeepAlive(tx)
}

// ExecuteContext runs a transaction query with cancellation and execution limits.
func (tx *Transaction) ExecuteContext(ctx context.Context, query string, params map[string]any, options *ExecutionOptions) (*QueryResult, error) {
	paramsJSON, err := marshalExecutionParams(params)
	if err != nil {
		return nil, err
	}
	return tx.executeContextJSON(ctx, query, paramsJSON, options)
}
func (tx *Transaction) executeContextJSON(ctx context.Context, query, paramsJSON string, options *ExecutionOptions) (*QueryResult, error) {
	if err := validateExecutionStrings(query, paramsJSON); err != nil {
		return nil, err
	}
	if err := tx.acquire(); err != nil {
		return nil, err
	}
	defer tx.release()
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
	r := C.grafeo_transaction_execute_with_options(tx.handle, cQuery, cParams, &inv.options)
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

// Execute runs a query within this transaction.
func (tx *Transaction) Execute(query string) (*QueryResult, error) {
	return tx.executeContextJSON(context.Background(), query, "", nil)
}

// ExecuteParams runs a transaction query with parameters as a Go map.
func (tx *Transaction) ExecuteParams(query string, params map[string]any) (*QueryResult, error) {
	return tx.ExecuteContext(context.Background(), query, params, nil)
}

// ExecuteWithParams runs a transaction query with JSON object parameters.
func (tx *Transaction) ExecuteWithParams(query string, paramsJSON string) (*QueryResult, error) {
	return tx.executeContextJSON(context.Background(), query, paramsJSON, nil)
}

// ExecuteLanguage runs a transaction query in the given language with optional
// JSON object parameters. Pass an empty paramsJSON when no parameters are needed.
func (tx *Transaction) ExecuteLanguage(language, query, paramsJSON string) (*QueryResult, error) {
	return tx.executeContextJSON(context.Background(), query, paramsJSON, &ExecutionOptions{Language: language})
}

// Commit commits the transaction and releases its native handle on success.
func (tx *Transaction) Commit() error { return tx.complete(true) }

// Rollback aborts the transaction and releases its native handle on success.
func (tx *Transaction) Rollback() error { return tx.complete(false) }

func (tx *Transaction) complete(commit bool) error {
	if err := tx.acquire(); err != nil {
		return err
	}
	defer tx.release()
	runtime.LockOSThread()
	var status C.GrafeoStatus
	if commit {
		status = C.grafeo_commit(tx.handle)
	} else {
		status = C.grafeo_rollback(tx.handle)
	}
	err := statusToError(status)
	runtime.UnlockOSThread()
	if err != nil {
		return err
	}
	C.grafeo_free_transaction(tx.handle)
	tx.handle = nil
	runtime.SetFinalizer(tx, nil)
	return nil
}

// free is the GC finalizer: native free automatically rolls back unfinished work.
func (tx *Transaction) free() {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	if tx.handle != nil {
		tx.db.mu.RLock()
		defer tx.db.release()
		// Retain the parent while native free releases the active transaction lease.
		C.grafeo_free_transaction(tx.handle)
		tx.handle = nil
		runtime.KeepAlive(tx.db)
	}
}
