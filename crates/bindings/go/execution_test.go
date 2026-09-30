package grafeo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"
)

func executionDB(t *testing.T) *Database {
	t.Helper()
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func errorCode(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected %s, got nil", want)
	}
	var typed *Error
	if !errors.As(err, &typed) {
		t.Fatalf("expected *Error, got %T: %v", err, err)
	}
	if typed.Code != want {
		t.Fatalf("error code = %q, want %q", typed.Code, want)
	}
}

func TestQueryControlIsSingleUseAndPreCancelDoesNotMutate(t *testing.T) {
	db := executionDB(t)
	control, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	control.Cancel()
	_, err = db.ExecuteContext(context.Background(),
		"INSERT (:Cancelled {n: 1}) RETURN 1", nil, &ExecutionOptions{Control: control})
	errorCode(t, err, "GRAFEO-Q007")
	if result, _ := db.Execute("MATCH (n:Cancelled) RETURN n"); len(result.Rows) != 0 {
		t.Fatalf("cancelled mutation created %d rows", len(result.Rows))
	}
	_, err = db.ExecuteContext(context.Background(), "RETURN 1", nil, &ExecutionOptions{Control: control})
	if err == nil {
		t.Fatal("reusing consumed control succeeded")
	}
	fresh, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = db.ExecuteContext(context.Background(), "RETURN 1", nil, &ExecutionOptions{Control: fresh}); err != nil {
		t.Fatal(err)
	}
}

func TestQueryControlTimeoutsAreTyped(t *testing.T) {
	for _, timeout := range []time.Duration{0, -1} {
		control, err := NewQueryControl(timeout)
		if err != nil {
			t.Fatal(err)
		}
		if timeout < 0 {
			if err := control.Close(); err != nil {
				t.Fatal(err)
			}
			continue
		}
		db := executionDB(t)
		_, err = db.ExecuteContext(context.Background(), "RETURN 1", nil, &ExecutionOptions{Control: control})
		errorCode(t, err, "GRAFEO-Q003")
	}
}

func TestContextCancellationIsTypedAndIsolated(t *testing.T) {
	db := executionDB(t)
	if _, err := db.Execute("UNWIND range(1, 300) AS i INSERT (:CancelWork {i: i})"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := db.ExecuteStreamContext(ctx, "MATCH (a:CancelWork), (b:CancelWork), (c:CancelWork) WHERE a.i + b.i + c.i < 0 RETURN a.i", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := stream.Next(); done <- err }()
	deadline := time.Now().Add(5 * time.Second)
	for stream.active.Load() == 0 && time.Now().Before(deadline) {
		runtime.Gosched()
	}
	if stream.active.Load() == 0 {
		t.Fatal("pull never became active")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("context cancellation did not stop native pull")
	}
	if err := stream.Close(); !errors.Is(err, context.Canceled) {
		t.Fatalf("sticky close: %v", err)
	}
	if _, err := db.Execute("RETURN 1"); err != nil {
		t.Fatal(err)
	}
}

func TestLimitsRollbackMutationAndPreservePriorTransactionWrite(t *testing.T) {
	db := executionDB(t)
	zero := uint64(0)
	_, err := db.ExecuteContext(context.Background(), "INSERT (:Denied) RETURN 1", nil, &ExecutionOptions{MaxRows: &zero})
	if err == nil {
		t.Fatal("zero row cap admitted mutation")
	}
	if result, _ := db.Execute("MATCH (n:Denied) RETURN n"); len(result.Rows) != 0 {
		t.Fatalf("denied mutation created %d rows", len(result.Rows))
	}
	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tx.ExecuteContext(context.Background(), "INSERT (:Kept {n: 1})", nil, nil); err != nil {
		t.Fatal(err)
	}
	cap := uint64(1)
	_, err = tx.ExecuteContext(context.Background(), "INSERT (:Denied {payload: 'large'}) RETURN 1", nil, &ExecutionOptions{MaxBytes: &cap})
	if err == nil {
		t.Fatal("byte cap admitted denied transaction statement")
	}
	if err = tx.Commit(); err != nil {
		t.Fatal(err)
	}
	result, err := db.Execute("MATCH (n:Kept) RETURN n")
	if err != nil || len(result.Rows) != 1 {
		t.Fatalf("prior transaction write not preserved: rows=%v err=%v", result, err)
	}
}

func TestStreamChunksAndClose(t *testing.T) {
	db := executionDB(t)
	stream, err := db.ExecuteStreamContext(context.Background(), "UNWIND range(1, 2500) AS i RETURN i", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	count := 0
	seen := make(map[int64]bool)
	for {
		chunk, err := stream.NextChunk(257)
		if err != nil {
			t.Fatal(err)
		}
		if chunk == nil {
			break
		}
		if len(chunk.Rows) > 257 {
			t.Fatalf("chunk exceeded requested rows: %d", len(chunk.Rows))
		}
		for _, row := range chunk.Rows {
			v, e := row["i"].(json.Number).Int64()
			if e != nil || seen[v] || v < 1 || v > 2500 {
				t.Fatalf("invalid duplicate row: %v", row)
			}
			seen[v] = true
		}
		count += len(chunk.Rows)
	}
	if count != 2500 {
		t.Fatalf("chunked row count = %d, want 2500", count)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestStreamEarlyCloseAndConcurrentCancel(t *testing.T) {
	db := executionDB(t)
	control, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := db.ExecuteStreamContext(context.Background(),
		"UNWIND range(1, 2500) AS i RETURN i", nil, &ExecutionOptions{Control: control})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = stream.Next(); err != nil {
		t.Fatal(err)
	}
	closed := make(chan error, 1)
	go func() { closed <- stream.Close() }()
	control.Cancel()
	firstClose := <-closed
	if firstClose != nil && !errors.Is(firstClose, context.Canceled) {
		t.Fatal(firstClose)
	}
	if secondClose := stream.Close(); (firstClose == nil) != (secondClose == nil) {
		t.Fatalf("close outcome changed: %v then %v", firstClose, secondClose)
	}
	if _, err = db.Execute("RETURN 1"); err != nil {
		t.Fatal(fmt.Errorf("database poisoned after stream close: %w", err))
	}
}

func TestStreamLimitsFailWithoutTruncation(t *testing.T) {
	db := executionDB(t)
	maxRows := uint64(1)
	stream, err := db.ExecuteStreamContext(context.Background(),
		"UNWIND [1, 2] AS i RETURN i", nil, &ExecutionOptions{MaxRows: &maxRows})
	if err != nil {
		t.Fatal(err)
	}
	if row, err := stream.Next(); err != nil || row == nil {
		t.Fatalf("first row: row=%v err=%v", row, err)
	}
	if _, err = stream.Next(); err == nil {
		t.Fatal("stream silently truncated at row cap")
	}
	errorCode(t, stream.Close(), "GRAFEO-S001")
}

func TestIndependentStreamsCancelInIsolation(t *testing.T) {
	db := executionDB(t)
	one, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	two, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	first, err := db.ExecuteStreamContext(context.Background(),
		"UNWIND range(1, 2500) AS i RETURN i", nil, &ExecutionOptions{Control: one})
	if err != nil {
		t.Fatal(err)
	}
	second, err := db.ExecuteStreamContext(context.Background(),
		"UNWIND range(1, 2500) AS i RETURN i", nil, &ExecutionOptions{Control: two})
	if err != nil {
		t.Fatal(err)
	}
	one.Cancel()
	if _, err = first.Next(); err == nil {
		t.Fatal("cancelled stream produced a row")
	}
	count := 0
	for {
		row, nextErr := second.Next()
		if nextErr != nil {
			t.Fatal(nextErr)
		}
		if row == nil {
			break
		}
		count++
	}
	if count != 2500 {
		t.Fatalf("independent stream produced %d rows", count)
	}
	errorCode(t, first.Close(), "GRAFEO-Q007")
	if err = second.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestBusyParentCloseRetainsHandleAndCompletedTransactionIsClosed(t *testing.T) {
	db := executionDB(t)
	tx, err := db.BeginTransaction()
	if err != nil {
		t.Fatal(err)
	}
	if err = db.Close(); err == nil {
		t.Fatal("parent closed with active transaction")
	}
	if _, err = tx.Execute("INSERT (:Retained)"); err != nil {
		t.Fatal(err)
	}
	if err = tx.Commit(); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Execute("RETURN 1"); !errors.Is(err, ErrClosed) {
		t.Fatalf("completed tx: %v", err)
	}
	stream, err := db.ExecuteStream("MATCH (n:Retained) RETURN n")
	if err != nil {
		t.Fatal(err)
	}
	if err = db.Close(); err == nil {
		t.Fatal("parent closed with active stream")
	}
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err = db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = db.Execute("RETURN 1"); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed db: %v", err)
	}
}

func TestCollectFailureIsBoundedAndSticky(t *testing.T) {
	db := executionDB(t)
	bytes := uint64(16000)
	stream, err := db.ExecuteStreamContext(context.Background(), "UNWIND range(1,1000) AS i RETURN i", nil, &ExecutionOptions{MaxBytes: &bytes})
	if err != nil {
		t.Fatal(err)
	}
	rows, err := stream.Collect()
	if rows != nil {
		t.Fatal("collection exposed partial rows")
	}
	errorCode(t, err, "GRAFEO-S001")
	errorCode(t, stream.Close(), "GRAFEO-S001")
	_, err = stream.Next()
	errorCode(t, err, "GRAFEO-S001")
}

func TestDatabaseCloseAndCRUDConcurrentLifetime(t *testing.T) {
	db := executionDB(t)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			_, err := db.CreateNode([]string{"Race"}, nil)
			if err != nil && !errors.Is(err, ErrClosed) && !errors.Is(err, ErrBusy) {
				t.Errorf("CRUD: %v", err)
				return
			}
		}
	}()
	for i := 0; i < 100; i++ {
		err := db.Close()
		if err == nil {
			break
		}
		if !errors.Is(err, ErrBusy) {
			t.Fatalf("close: %v", err)
		}
		runtime.Gosched()
	}
	<-done
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCopiedGoResultDenialPrecedesCommit(t *testing.T) {
	db := executionDB(t)
	cap := uint64(16000)
	_, err := db.ExecuteContext(context.Background(), "INSERT (:CopyDenied) RETURN $payload AS payload", map[string]any{"payload": strings.Repeat("x", 1000)}, &ExecutionOptions{MaxBytes: &cap})
	errorCode(t, err, "GRAFEO-S001")
	result, err := db.Execute("MATCH (n:CopyDenied) RETURN n")
	if err != nil || len(result.Rows) != 0 {
		t.Fatalf("copy denial committed: result=%v err=%v", result, err)
	}
}

func TestContextDeadlineWithExplicitControlPreservesDeadlineCategory(t *testing.T) {
	db := executionDB(t)
	if _, err := db.Execute("UNWIND range(1,300) AS i INSERT (:DeadlineWork {i:i})"); err != nil {
		t.Fatal(err)
	}
	control, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	defer control.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	stream, err := db.ExecuteStreamContext(ctx, "MATCH (a:DeadlineWork), (b:DeadlineWork), (c:DeadlineWork) WHERE a.i+b.i+c.i < 0 RETURN a.i", nil, &ExecutionOptions{Control: control})
	if err == nil {
		_, err = stream.Next()
		defer stream.Close()
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline category lost: %v", err)
	}
	var native *Error
	if !errors.As(err, &native) || (native.Code != "GRAFEO-Q003" && native.Code != "GRAFEO-Q007") {
		t.Fatalf("native deadline code lost: %v", err)
	}
}

func TestSharedCopyEnvelopeDenialsNeverFollowCommit(t *testing.T) {
	payloads := []any{strings.Repeat("\x01é", 100), []any{1.234567890123456e200, true, nil, "nested"}, map[string]any{"a": strings.Repeat("x", 100), "b": []any{1, 2, 3}}}
	for shape, payload := range payloads {
		accepted := false
		for _, cap := range []uint64{4096, 16384, 65536, 262144} {
			db := executionDB(t)
			result, err := db.ExecuteContext(context.Background(), "INSERT (:BudgetShape) RETURN $payload AS payload", map[string]any{"payload": payload}, &ExecutionOptions{MaxBytes: &cap})
			check, checkErr := db.Execute("MATCH (n:BudgetShape) RETURN n")
			if checkErr != nil {
				t.Fatal(checkErr)
			}
			if err != nil {
				errorCode(t, err, "GRAFEO-S001")
				if len(check.Rows) != 0 {
					t.Fatalf("shape%d cap%d failed after commit: %v", shape, cap, err)
				}
			} else {
				accepted = true
				if len(result.Rows) != 1 || len(check.Rows) != 1 {
					t.Fatalf("shape%d success mismatch", shape)
				}
			}
		}
		if !accepted {
			t.Fatalf("shape%d never admitted", shape)
		}
	}
}

func TestControlCloseRacesOnlyIndependentInvocationHandles(t *testing.T) {
	db := executionDB(t)
	control, err := NewQueryControl(-1)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := db.ExecuteStreamContext(ctx, "UNWIND range(1,2500) AS i RETURN i", nil, &ExecutionOptions{Control: control})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			_ = control.Cancel()
		}
	}()
	if err = control.Close(); err != nil {
		t.Fatal(err)
	}
	cancel()
	<-done
	_, err = stream.Next()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("invocation lost retained authority: %v", err)
	}
	if err = stream.Close(); !errors.Is(err, context.Canceled) {
		t.Fatalf("close lost error: %v", err)
	}
}
