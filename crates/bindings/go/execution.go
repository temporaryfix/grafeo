package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
	"unsafe"
)

// ExecutionOptions supplies one execution owner and explicit result limits.
// Nil limits use 1,000,000 rows and 64 MiB for eager results. Streaming has no
// total row limit unless MaxRows is set. MaxBytes is the combined native and Go
// copy envelope, applied per delivered row/chunk for streaming. One quarter is
// reserved for native admission before execution commits; the remainder bounds
// Go conversion. A pointer to zero is a real zero limit. Language defaults to GQL.
type ExecutionOptions struct {
	Control  *QueryControl
	MaxRows  *uint64
	MaxBytes *uint64
	Language string
}

// QueryControl is a single-use execution owner with independent cancellation.
// Cancel and Close may run concurrently with execution. Close frees this Go
// owner's native handles without invalidating an execution already started.
type QueryControl struct {
	mu           sync.Mutex
	owner        *C.GrafeoQueryControl
	cancellation *C.GrafeoCancelHandle
	consumed     bool
	closed       bool
}

// NewQueryControl starts the timeout now. Negative durations mean no deadline;
// zero is an immediate deadline. Positive durations round up to milliseconds.
func NewQueryControl(timeout time.Duration) (*QueryControl, error) {
	ms := int64(-1)
	if timeout >= 0 {
		ms = int64(timeout / time.Millisecond)
		if timeout%time.Millisecond != 0 {
			ms++
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	owner := C.grafeo_query_control_create(C.int64_t(ms))
	if owner == nil {
		return nil, lastError()
	}
	cancellation := C.grafeo_query_control_cancel_handle(owner)
	if cancellation == nil {
		err := lastError()
		C.grafeo_query_control_free(owner)
		return nil, err
	}
	control := &QueryControl{owner: owner, cancellation: cancellation}
	runtime.SetFinalizer(control, (*QueryControl).finalize)
	return control, nil
}

func (q *QueryControl) finalize() { _ = q.Close() }

// Cancel requests cancellation; calls on a closed control return ErrClosed.
func (q *QueryControl) Cancel() error {
	if q == nil {
		return ErrClosed
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	defer runtime.KeepAlive(q)
	if q.closed || q.cancellation == nil {
		return ErrClosed
	}
	return lockAndCheckStatus(func() C.GrafeoStatus { return C.grafeo_cancel(q.cancellation) })
}

// Close is idempotent and does not cancel an execution that consumed the owner.
func (q *QueryControl) Close() error {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return nil
	}
	q.closed = true
	C.grafeo_query_control_free(q.owner)
	C.grafeo_cancel_handle_free(q.cancellation)
	q.owner = nil
	q.cancellation = nil
	runtime.SetFinalizer(q, nil)
	return nil
}

// begin reserves the single use and transfers the native owner into invocation.
// Its cloned cancellation allocation remains valid even if q is closed.
func (q *QueryControl) begin() (*C.GrafeoQueryControl, *C.GrafeoCancelHandle, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	defer runtime.KeepAlive(q)
	if q.closed {
		return nil, nil, ErrClosed
	}
	if q.consumed {
		return nil, nil, invalidExecution("query control has already been consumed")
	}
	if q.owner == nil || q.cancellation == nil {
		return nil, nil, invalidExecution("query control was not initialized")
	}
	runtime.LockOSThread()
	clone := C.grafeo_cancel_handle_clone(q.cancellation)
	if clone == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, nil, err
	}
	runtime.UnlockOSThread()
	q.consumed = true
	owner := q.owner
	q.owner = nil
	return owner, clone, nil
}

type invocation struct {
	options      C.GrafeoQueryOptions
	maxBytes     uint64
	ctx          context.Context
	mu           sync.Mutex
	cancellation *C.GrafeoCancelHandle
	stop         func() bool
	callbackDone chan struct{}
	once         sync.Once
}

func newInvocation(ctx context.Context, options *ExecutionOptions, streaming bool) (*invocation, error) {
	if ctx == nil {
		return nil, invalidExecution("nil context")
	}
	rows, bytes := uint64(1_000_000), uint64(64<<20)
	if streaming {
		rows = uint64(^C.size_t(0))
	}
	var control *QueryControl
	var language string
	if options != nil {
		control, language = options.Control, options.Language
		if options.MaxRows != nil {
			rows = *options.MaxRows
		}
		if options.MaxBytes != nil {
			bytes = *options.MaxBytes
		}
	}
	if strings.IndexByte(language, 0) >= 0 || !utf8.ValidString(language) {
		return nil, invalidExecution("language must be valid UTF-8 without NUL")
	}
	if uint64(C.size_t(rows)) != rows || uint64(C.size_t(bytes)) != bytes {
		return nil, invalidExecution("result limits exceed native size_t range")
	}
	if control == nil {
		timeout := time.Duration(-1)
		if deadline, ok := ctx.Deadline(); ok {
			timeout = time.Until(deadline)
			if timeout < 0 {
				timeout = 0
			}
		}
		var err error
		control, err = NewQueryControl(timeout)
		if err != nil {
			return nil, err
		}
		defer control.Close()
	}
	owner, cancellation, err := control.begin()
	if err != nil {
		return nil, err
	}
	// Native admission runs before an eager mutation commits. Its conservative
	// JSON estimate also bounds the later Go scanner within the reserved 3:1
	// conversion allowance, so successful native admission cannot defer an
	// ordinary copy-budget rejection until after commit.
	nativeBytes := bytes / 4
	i := &invocation{
		options:  C.GrafeoQueryOptions{control: owner, max_rows: C.size_t(rows), max_bytes: C.size_t(nativeBytes)},
		maxBytes: bytes - nativeBytes, ctx: ctx, cancellation: cancellation,
		callbackDone: make(chan struct{}),
	}
	if language != "" {
		i.options.language = C.CString(language)
	}
	i.stop = context.AfterFunc(ctx, func() { defer close(i.callbackDone); i.cancel() })
	// AfterFunc starts asynchronously: an already cancelled context must cancel
	// the native owner before the caller can begin its query.
	if ctx.Err() != nil {
		i.cancel()
	}
	return i, nil
}

// cancel only touches an independently retained native cancellation handle.
func (i *invocation) cancel() {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.cancellation != nil {
		C.grafeo_cancel(i.cancellation)
	}
}

// finish joins an already-started callback before releasing native allocations.
func (i *invocation) finish() {
	if i == nil {
		return
	}
	i.once.Do(func() {
		if !i.stop() {
			<-i.callbackDone
		}
		i.mu.Lock()
		defer i.mu.Unlock()
		C.grafeo_cancel_handle_free(i.cancellation)
		i.cancellation = nil
		C.grafeo_query_control_free(i.options.control)
		i.options.control = nil
		C.free(unsafe.Pointer(i.options.language))
		i.options.language = nil
	})
}

// error keeps native codes intact while exposing context deadline cancellation.
func (i *invocation) error(err error) error {
	var native *Error
	if errors.As(err, &native) && native.Code == "GRAFEO-Q007" && errors.Is(i.ctx.Err(), context.DeadlineExceeded) {
		copy := *native
		copy.Cause = context.DeadlineExceeded
		return &copy
	}
	return err
}
