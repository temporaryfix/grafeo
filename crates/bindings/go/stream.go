package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"context"
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
	"unsafe"
)

// ResultStream pulls bounded native rows or chunks on demand. Always Close it
// to release query resources. Unsupported operators return a native error.
// Concurrent pulls serialize; Close interrupts an active pull before joining it.
type ResultStream struct {
	mu          sync.Mutex
	handle      *C.GrafeoStream
	columns     []string
	invocation  *invocation
	database    *Database
	maxBytes    uint64
	collectRows uint64
	terminal    bool
	closed      bool
	failure     error
	closing     atomic.Bool
	active      atomic.Int64
}

var ErrStreamClosed = errors.New("grafeo: stream is closed")

func (db *Database) ExecuteStream(query string) (*ResultStream, error) {
	return db.ExecuteStreamContext(context.Background(), query, nil, nil)
}

// ExecuteStreamContext keeps context cancellation live until EOF or Close.
// MaxBytes bounds each native/Go copy; MaxRows optionally limits total rows.
func (db *Database) ExecuteStreamContext(ctx context.Context, query string, params map[string]any, options *ExecutionOptions) (*ResultStream, error) {
	if strings.IndexByte(query, 0) >= 0 || !utf8.ValidString(query) {
		return nil, invalidExecution("query contains NUL")
	}
	var data []byte
	var err error
	if params != nil {
		data, err = json.Marshal(params)
		if err != nil {
			return nil, err
		}
	}
	if err = db.acquire(); err != nil {
		return nil, err
	}
	defer db.release()
	inv, err := newInvocation(ctx, options, true)
	if err != nil {
		return nil, err
	}
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	var cParams *C.char
	if data != nil {
		cParams = C.CString(string(data))
		defer C.free(unsafe.Pointer(cParams))
	}
	runtime.LockOSThread()
	handle := C.grafeo_stream_open_with_options(db.handle, cQuery, cParams, &inv.options)
	if handle == nil {
		err = inv.error(lastError())
		runtime.UnlockOSThread()
		inv.finish()
		return nil, err
	}
	runtime.UnlockOSThread()
	cols, err := readStreamColumns(handle, inv.maxBytes)
	if err != nil {
		runtime.LockOSThread()
		cleanup := statusToError(C.grafeo_stream_close(handle))
		C.grafeo_stream_free(handle)
		runtime.UnlockOSThread()
		inv.finish()
		return nil, errors.Join(err, cleanup)
	}
	rows := uint64(1_000_000)
	if options != nil && options.MaxRows != nil {
		rows = *options.MaxRows
	}
	stream := &ResultStream{handle: handle, columns: cols, invocation: inv, database: db, maxBytes: inv.maxBytes, collectRows: rows}
	runtime.SetFinalizer(stream, (*ResultStream).finalize)
	return stream, nil
}

// Columns returns an owned copy of the immutable schema.
func (s *ResultStream) Columns() []string { return append([]string(nil), s.columns...) }

func (s *ResultStream) finishLocked(primary error) error {
	if !s.terminal {
		s.terminal = true
		runtime.LockOSThread()
		cleanup := statusToError(C.grafeo_stream_close(s.handle))
		C.grafeo_stream_free(s.handle)
		runtime.UnlockOSThread()
		s.handle = nil
		if primary != nil {
			s.failure = primary
		} else if cleanup != nil {
			s.failure = s.invocation.error(cleanup)
		}
		if primary != nil && cleanup != nil && primary.Error() != cleanup.Error() {
			s.failure = errors.Join(primary, s.invocation.error(cleanup))
		}
		s.invocation.finish()
		s.database = nil
	}
	return s.failure
}

// Next returns nil,nil on clean exhaustion. Terminal failures remain observable.
func (s *ResultStream) Next() (map[string]any, error) {
	row, _, err := s.nextBounded(s.maxBytes)
	return row, err
}

func (s *ResultStream) nextBounded(budget uint64) (map[string]any, uint64, error) {
	s.active.Add(1)
	defer s.active.Add(-1)
	s.mu.Lock()
	defer s.mu.Unlock()
	defer runtime.KeepAlive(s)
	if s.failure != nil {
		return nil, 0, s.failure
	}
	if s.closed || s.closing.Load() {
		return nil, 0, ErrStreamClosed
	}
	if s.terminal {
		return nil, 0, s.failure
	}
	var ptr *C.char
	runtime.LockOSThread()
	status := C.grafeo_stream_next_row_json(s.handle, &ptr)
	if status != C.GRAFEO_OK {
		err := s.invocation.error(lastError())
		runtime.UnlockOSThread()
		return nil, 0, s.finishLocked(err)
	}
	runtime.UnlockOSThread()
	if ptr == nil {
		return nil, 0, s.finishLocked(nil)
	}
	defer C.grafeo_free_string(ptr)
	cost, err := nativeJSONCopyCost(ptr, budget)
	if err != nil {
		return nil, 0, s.finishLocked(err)
	}
	raw, err := boundedNativeString(ptr, budget)
	if err != nil {
		return nil, 0, s.finishLocked(err)
	}
	var row map[string]any
	decoder := json.NewDecoder(strings.NewReader(raw))
	if err = decoder.Decode(&row); err != nil {
		return nil, 0, s.finishLocked(err)
	}
	return row, cost, nil
}

// NextChunk returns at most maxRows (capped natively at 1024) and MaxBytes.
// Row and chunk pulls share the same cursor. A nil result denotes EOF.
func (s *ResultStream) NextChunk(maxRows int) (*QueryResult, error) {
	if maxRows <= 0 {
		return nil, invalidExecution("chunk row count must be positive")
	}
	s.active.Add(1)
	defer s.active.Add(-1)
	s.mu.Lock()
	defer s.mu.Unlock()
	defer runtime.KeepAlive(s)
	if s.failure != nil {
		return nil, s.failure
	}
	if s.closed || s.closing.Load() {
		return nil, ErrStreamClosed
	}
	if s.terminal {
		return nil, s.failure
	}
	var result *C.GrafeoResult
	runtime.LockOSThread()
	status := C.grafeo_stream_next_chunk(s.handle, C.size_t(maxRows), &result)
	if status != C.GRAFEO_OK {
		err := s.invocation.error(lastError())
		runtime.UnlockOSThread()
		return nil, s.finishLocked(err)
	}
	runtime.UnlockOSThread()
	if result == nil {
		return nil, s.finishLocked(nil)
	}
	defer C.grafeo_free_result(result)
	parsed, err := parseResultBounded(result, s.maxBytes)
	if err != nil {
		return nil, s.finishLocked(err)
	}
	return parsed, nil
}

// Collect returns the remaining rows subject to total row and copied-byte caps.
// Failure returns no partial result. Prefer Next/NextChunk for larger streams.
func (s *ResultStream) Collect() ([]map[string]any, error) {
	var rows []map[string]any
	remaining := s.maxBytes
	fail := func(message string) ([]map[string]any, error) {
		s.mu.Lock()
		defer s.mu.Unlock()
		err := copyLimitError(message)
		if s.terminal {
			if s.failure == nil {
				s.failure = err
			}
			return nil, s.failure
		}
		return nil, s.finishLocked(err)
	}
	for {
		row, cost, err := s.nextBounded(remaining)
		if err != nil {
			return nil, err
		}
		if row == nil {
			return rows, nil
		}
		if uint64(len(rows)) >= s.collectRows {
			return fail("collection exceeds row limit")
		}
		remaining -= cost
		if len(rows) == cap(rows) {
			next := cap(rows) * 2
			if next < 8 {
				next = 8
			}
			// Reserve the complete new allocation while the old slice remains live.
			extra := uint64(next) * uint64(unsafe.Sizeof(row))
			if extra > remaining {
				return fail("collection capacity exceeds byte limit")
			}
			remaining -= extra
			grown := make([]map[string]any, len(rows), next)
			copy(grown, rows)
			rows = grown
		}
		rows = append(rows, row)
	}
}

// Close is idempotent and fallible. An active pull is cancelled, then joined
// before its native pointer and cancellation allocations are freed.
func (s *ResultStream) Close() error {
	s.closing.Store(true)
	if s.active.Load() != 0 {
		s.invocation.cancel()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.failure
	}
	s.closed = true
	err := s.finishLocked(nil)
	runtime.SetFinalizer(s, nil)
	return err
}
func (s *ResultStream) finalize() { _ = s.Close() }

func readStreamColumns(handle *C.GrafeoStream, maxBytes uint64) ([]string, error) {
	runtime.LockOSThread()
	ptr := C.grafeo_stream_columns_json(handle)
	if ptr == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_string(ptr)
	raw, err := boundedNativeString(ptr, maxBytes)
	if err != nil {
		return nil, err
	}
	var columns []string
	if err = json.Unmarshal([]byte(raw), &columns); err != nil {
		return nil, err
	}
	return columns, nil
}
