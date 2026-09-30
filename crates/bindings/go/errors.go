package grafeo

/*
#include "grafeo.h"
*/
import "C"
import (
	"context"
	"errors"
	"fmt"
	"runtime"
)

// ErrDatabase is the base error for all Grafeo database errors.
var ErrDatabase = errors.New("grafeo")

// ErrBusy means an operation could not acquire an active handle.
var ErrBusy = fmt.Errorf("%w: handle is busy", ErrDatabase)

// ErrClosed means the handle has already been closed.
var ErrClosed = fmt.Errorf("%w: handle is closed", ErrDatabase)

// Error preserves the native structured code and its Go cancellation category.
type Error struct {
	Code    string
	Message string
	Cause   error
}

func (e *Error) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("grafeo: %s: %s", e.Code, e.Message)
	}
	return "grafeo: " + e.Message
}

// Unwrap preserves both the database family and context cancellation identity.
func (e *Error) Unwrap() []error {
	if e.Cause != nil {
		return []error{ErrDatabase, e.Cause}
	}
	return []error{ErrDatabase}
}

func invalidExecution(message string) error { return &Error{Message: message} }

// lastError reads the thread-local error from the C layer.
// Must be called on the same OS thread as the C call that set the error.
func lastError() error {
	msg := C.grafeo_last_error()
	err := &Error{Message: "unknown error"}
	if msg != nil {
		err.Message = C.GoString(msg)
	}
	if code := C.grafeo_last_error_code(); code != nil {
		err.Code = C.GoString(code)
	}
	switch err.Code {
	case "GRAFEO-Q007":
		err.Cause = context.Canceled
	case "GRAFEO-Q003":
		err.Cause = context.DeadlineExceeded
	}
	return err
}

// statusToError converts a GrafeoStatus to a Go error (nil on success).
// Must be called on the same OS thread as the C call that produced the status.
func statusToError(status C.GrafeoStatus) error {
	if status == C.GRAFEO_OK {
		return nil
	}
	return lastError()
}

// lockAndCheckStatus pins the goroutine to an OS thread, calls fn,
// and reads any error from the thread-local. This ensures the C call
// and error retrieval happen on the same OS thread.
func lockAndCheckStatus(fn func() C.GrafeoStatus) error {
	runtime.LockOSThread()
	status := fn()
	err := statusToError(status)
	runtime.UnlockOSThread()
	return err
}
