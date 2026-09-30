package grafeo

/*
#include "grafeo.h"
#include <string.h>
*/
import "C"
import (
	"encoding/json"
	"strings"
	"unsafe"
)

// QueryResult holds the result of a query execution.
type QueryResult struct {
	Columns         []string
	Rows            []Row
	ExecutionTimeMs float64
	RowsScanned     uint64
}

// Row is a single result row, mapping column names to values.
type Row map[string]any

// Node represents a graph node with labels and properties.
type Node struct {
	ID         uint64
	Labels     []string
	Properties map[string]any
}

// Edge represents a graph edge between two nodes.
type Edge struct {
	ID         uint64
	SourceID   uint64
	TargetID   uint64
	Type       string
	Properties map[string]any
}

// IsolationLevel controls transaction isolation.
type IsolationLevel int32

const (
	// ReadCommitted sees only committed data but may see different versions
	// within a transaction.
	ReadCommitted IsolationLevel = 0

	// SnapshotIsolation (default) sees a consistent snapshot as of transaction start.
	SnapshotIsolation IsolationLevel = 1

	// Serializable provides full serializability via SSI conflict detection.
	Serializable IsolationLevel = 2
)

// VectorResult holds a single nearest-neighbor search result.
type VectorResult struct {
	NodeID   uint64
	Distance float32
}

// IndexID is the committed catalog owner returned by CreateIndex.
type IndexID uint32

// IndexKind identifies the requested index family.
type IndexKind uint32

const (
	PropertyIndex IndexKind = iota
	BTreeIndex
	TextIndex
	VectorIndex
)

// CreateIndexRequest selects an exact graph path and preserves option presence.
// Label is required for Text/Vector and absent for Property/BTree.
// Numeric pointers distinguish a supplied zero (validated by the engine) from absence.
type CreateIndexRequest struct {
	Graph          []string
	Name           *string
	Label          *string
	Property       string
	Kind           IndexKind
	Dimensions     *uint
	Metric         *string
	M              *uint
	EfConstruction *uint
	// MinTokenLength is Text-only; nil selects the default (2), zero is valid.
	MinTokenLength *uint
	Quantization   *string
}

// SearchOption configures vector search.
type SearchOption func(*searchConfig)

type searchConfig struct {
	ef int32
}

// WithEf sets the search ef parameter for recall/speed tradeoff.
func WithEf(ef int) SearchOption {
	return func(c *searchConfig) { c.ef = int32(ef) }
}

// parseResult converts a C GrafeoResult into a Go QueryResult.
func parseResult(r *C.GrafeoResult) (*QueryResult, error) {
	return parseResultBounded(r, 64<<20)
}

func parseResultBounded(r *C.GrafeoResult, maxBytes uint64) (*QueryResult, error) {
	jsonPtr := C.grafeo_result_json(r)
	if jsonPtr == nil {
		return &QueryResult{}, nil
	}
	if _, err := nativeJSONCopyCost(jsonPtr, maxBytes); err != nil {
		return nil, err
	}
	jsonStr, err := boundedNativeString(jsonPtr, maxBytes)
	if err != nil {
		return nil, err
	}

	var rawRows []map[string]any
	dec := json.NewDecoder(strings.NewReader(jsonStr))
	dec.UseNumber()
	if err := dec.Decode(&rawRows); err != nil {
		return nil, err
	}

	// Extract column names from first row, preserving JSON key order.
	// Go map iteration is random, so we parse the first row's keys manually
	// using json.Decoder to maintain the original order from the engine.
	var columns []string
	if len(rawRows) > 0 {
		columns = extractOrderedKeys([]byte(jsonStr))
	}

	rows := make([]Row, len(rawRows))
	for i, raw := range rawRows {
		rows[i] = Row(raw)
	}

	return &QueryResult{
		Columns:         columns,
		Rows:            rows,
		ExecutionTimeMs: float64(C.grafeo_result_execution_time_ms(r)),
		RowsScanned:     uint64(C.grafeo_result_rows_scanned(r)),
	}, nil
}

// extractOrderedKeys parses the first object in a JSON array and returns
// its keys in the order they appear, avoiding Go map's random iteration.
func extractOrderedKeys(data []byte) []string {
	dec := json.NewDecoder(strings.NewReader(string(data)))

	// Skip opening '['
	t, err := dec.Token()
	if err != nil || t != json.Delim('[') {
		return nil
	}

	// Skip opening '{'
	t, err = dec.Token()
	if err != nil || t != json.Delim('{') {
		return nil
	}

	var keys []string
	for dec.More() {
		t, err = dec.Token()
		if err != nil {
			break
		}
		if key, ok := t.(string); ok {
			keys = append(keys, key)
			// Skip the value
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				break
			}
		}
	}
	return keys
}

// C's precommit admission includes conservative JSON/decoded container costs.
// Check its serialized extent before copying a native allocation into Go.
func boundedNativeString(ptr *C.char, maxBytes uint64) (string, error) {
	if ptr == nil {
		return "", invalidExecution("null native JSON")
	}
	size := uint64(C.strlen(ptr))
	if size > maxBytes || size > uint64(^uint(0)>>1) {
		return "", copyLimitError("native JSON exceeds byte limit")
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(ptr)), int(size))), nil
}

func copyLimitError(message string) error { return &Error{Code: "GRAFEO-S001", Message: message} }

// Inspect borrowed, trusted native JSON before any Go string/decoder allocation.
// The envelope covers serialized copies, decoder buffers, map/slice spare slots,
// strings and number containers. C admission reserves a separate quarter budget.
func nativeJSONCopyCost(ptr *C.char, limit uint64) (uint64, error) {
	if ptr == nil {
		return 0, invalidExecution("null native JSON")
	}
	size := uint64(C.strlen(ptr))
	if size > limit || size > uint64(^uint(0)>>1) {
		return 0, copyLimitError("JSON exceeds copy envelope")
	}
	cost := uint64(0)
	charge := func(n uint64) bool {
		if n > limit-cost {
			return false
		}
		cost += n
		return true
	}
	for i := 0; i < 4; i++ {
		if !charge(size) {
			return 0, copyLimitError("JSON buffer copies exceed envelope")
		}
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), int(size))
	inString, escaped := false, false
	for _, b := range data {
		if inString {
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				inString = false
			}
			continue
		}
		switch b {
		case '"':
			inString = true
			if !charge(64) {
				return 0, copyLimitError("JSON string slots exceed envelope")
			}
		case '{':
			if !charge(576) {
				return 0, copyLimitError("JSON object exceeds envelope")
			}
		case ':':
			if !charge(320) {
				return 0, copyLimitError("JSON map entries exceed envelope")
			}
		case '[':
			if !charge(128) {
				return 0, copyLimitError("JSON array exceeds envelope")
			}
		case ',':
			if !charge(96) {
				return 0, copyLimitError("JSON element slots exceed envelope")
			}
		}
	}
	return cost, nil
}
