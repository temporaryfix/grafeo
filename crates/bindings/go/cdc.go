package grafeo

/*
#include "grafeo.h"
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"strings"
	"unsafe"
)

// ChangeEvent is one owned native event. Coordinates retain every bit of u64.
// Before/After numbers use json.Number; creation fields come from the event's
// commit, rather than a later lookup in the graph.
type ChangeEvent struct {
	EntityID         uint64         `json:"entity_id,string"`
	EntityType       string         `json:"entity_type"`
	Kind             string         `json:"kind"`
	Epoch            uint64         `json:"epoch,string"`
	Timestamp        uint64         `json:"timestamp,string"`
	GraphIncarnation *uint64        `json:"graph_incarnation,string"`
	Before           map[string]any `json:"before"`
	After            map[string]any `json:"after"`
	Labels           []string       `json:"labels"`
	EdgeType         *string        `json:"edge_type"`
	SourceID         *uint64        `json:"src_id,string"`
	TargetID         *uint64        `json:"dst_id,string"`
	LPGGraph         []string       `json:"lpg_graph"`
	TripleGraph      *string        `json:"triple_graph"`
	TripleSubject    *string        `json:"triple_subject"`
	TriplePredicate  *string        `json:"triple_predicate"`
	TripleObject     *string        `json:"triple_object"`
}

// ChangePage owns a bounded event slice and its canonical exclusive cursor.
// No native handle survives this return, so stopping early requires no cleanup.
// Empty pages can advance over filtered events; an unchanged Next means EOF.
type ChangePage struct {
	Events []ChangeEvent
	Next   []byte
}

// SetCDCEnabled controls capture for subsequent sessions.
func (db *Database) SetCDCEnabled(enabled bool) error {
	if err := db.acquire(); err != nil {
		return err
	}
	defer db.release()
	C.grafeo_set_cdc_enabled(db.handle, C.bool(enabled))
	return nil
}

// CDCEnabled reports whether new sessions capture native changes.
func (db *Database) CDCEnabled() (bool, error) {
	if err := db.acquire(); err != nil {
		return false, err
	}
	defer db.release()
	return bool(C.grafeo_is_cdc_enabled(db.handle)), nil
}

// ChangesAfter reads an owned bounded page. Nil starts at the retained floor;
// a non-nil cursor must contain exactly 97 bytes. Limits must be positive.
// MaxBytes counts native event encodings, excluding JSON/page envelopes.
func (db *Database) ChangesAfter(cursor []byte, maxEvents, maxBytes int) (*ChangePage, error) {
	return db.readChangePage(cursor, maxEvents, maxBytes, func(ptr *C.uint8_t, length, rows, bytes C.size_t) *C.GrafeoChangePage {
		return C.grafeo_changes_after(db.handle, ptr, length, rows, bytes)
	})
}

// NodeHistoryAfter reads indexed node history at or after sinceEpoch.
// Cursor, limits and ownership follow ChangesAfter. Use epoch zero for all history.
func (db *Database) NodeHistoryAfter(id, sinceEpoch uint64, cursor []byte, maxEvents, maxBytes int) (*ChangePage, error) {
	return db.readChangePage(cursor, maxEvents, maxBytes, func(ptr *C.uint8_t, length, rows, bytes C.size_t) *C.GrafeoChangePage {
		return C.grafeo_node_history_after(db.handle, C.uint64_t(id), C.uint64_t(sinceEpoch), ptr, length, rows, bytes)
	})
}

// EdgeHistoryAfter reads indexed edge history at or after sinceEpoch.
// Cursor, limits and ownership follow ChangesAfter. Use epoch zero for all history.
func (db *Database) EdgeHistoryAfter(id, sinceEpoch uint64, cursor []byte, maxEvents, maxBytes int) (*ChangePage, error) {
	return db.readChangePage(cursor, maxEvents, maxBytes, func(ptr *C.uint8_t, length, rows, bytes C.size_t) *C.GrafeoChangePage {
		return C.grafeo_edge_history_after(db.handle, C.uint64_t(id), C.uint64_t(sinceEpoch), ptr, length, rows, bytes)
	})
}

func (db *Database) readChangePage(cursor []byte, maxEvents, maxBytes int,
	read func(*C.uint8_t, C.size_t, C.size_t, C.size_t) *C.GrafeoChangePage) (*ChangePage, error) {
	if err := db.acquire(); err != nil {
		return nil, err
	}
	defer db.release()
	// Negative Go bounds cannot be cast to unsigned C bounds. Zero exercises the
	// same native structured invalid-limit error without admitting a huge page.
	if maxEvents < 0 {
		maxEvents = 0
	}
	if maxBytes < 0 {
		maxBytes = 0
	}
	ptr := (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(cursor)))
	runtime.LockOSThread()
	page := read(ptr, C.size_t(len(cursor)), C.size_t(maxEvents), C.size_t(maxBytes))
	runtime.KeepAlive(cursor)
	if page == nil {
		err := lastError()
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()
	defer C.grafeo_free_change_page(page)

	result := &ChangePage{
		Next: C.GoBytes(unsafe.Pointer(C.grafeo_change_page_cursor(page)), C.GRAFEO_CDC_CURSOR_LEN),
	}
	decoder := json.NewDecoder(strings.NewReader(C.GoString(C.grafeo_change_page_events_json(page))))
	decoder.UseNumber()
	if err := decoder.Decode(&result.Events); err != nil {
		return nil, err
	}
	return result, nil
}
