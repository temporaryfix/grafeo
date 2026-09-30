package grafeo

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func cdcDB(t *testing.T) *Database {
	t.Helper()
	db, err := OpenInMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := db.SetCDCEnabled(true); err != nil {
		t.Fatal(err)
	}
	if enabled, err := db.CDCEnabled(); err != nil || !enabled {
		t.Fatalf("capture: %v %v", enabled, err)
	}
	return db
}

func cdcCode(t *testing.T, err error, code string) {
	t.Helper()
	var native *Error
	if !errors.As(err, &native) || native.Code != code {
		t.Fatalf("got %v; want %s", err, code)
	}
}

func TestCDCPagesOwnCompleteCreationEvents(t *testing.T) {
	db := cdcDB(t)
	a, err := db.CreateNode([]string{"First"}, map[string]any{"large": int64(9007199254740993)})
	if err != nil {
		t.Fatal(err)
	}
	b, err := db.CreateNode([]string{"Second"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	edge, err := db.CreateEdge(a.ID, b.ID, "LINK", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.SetCDCEnabled(false); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateNode([]string{"Uncaptured"}, nil); err != nil {
		t.Fatal(err)
	}
	var cursor []byte
	var events []ChangeEvent
	for _, id := range []uint64{a.ID, b.ID, edge.ID} {
		page, err := db.ChangesAfter(cursor, 1, 4096)
		if err != nil {
			t.Fatal(err)
		}
		if len(page.Events) != 1 || page.Events[0].EntityID != id || len(page.Next) != 97 {
			t.Fatalf("bad page: %+v", page)
		}
		cursor = page.Next
		events = append(events, page.Events...)
	}
	eof, err := db.ChangesAfter(cursor, 1, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if len(eof.Events) != 0 || !bytes.Equal(cursor, eof.Next) {
		t.Fatalf("bad EOF: %+v", eof)
	}
	nodePage, err := db.NodeHistoryAfter(a.ID, 0, nil, 1, 4096)
	if err != nil || len(nodePage.Events) != 1 {
		t.Fatalf("node page: %+v %v", nodePage, err)
	}
	filtered, err := db.NodeHistoryAfter(a.ID, nodePage.Events[0].Epoch+1, nil, 1, 1)
	if err != nil || len(filtered.Events) != 0 {
		t.Fatalf("epoch filtering before byte accounting: %+v %v", filtered, err)
	}
	edgePage, err := db.EdgeHistoryAfter(edge.ID, 0, nil, 1, 4096)
	if err != nil || len(edgePage.Events) != 1 {
		t.Fatalf("edge page: %+v %v", edgePage, err)
	}
	if _, err := db.NodeHistoryAfter(^uint64(0)-1, 0, nil, 1, 4096); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events[0].Labels, []string{"First"}) || events[0].After["large"] != json.Number("9007199254740993") {
		t.Fatalf("lost creation data: %+v", events[0])
	}
	e := edgePage.Events[0]
	if e.EdgeType == nil || *e.EdgeType != "LINK" || e.SourceID == nil || *e.SourceID != a.ID || e.TargetID == nil || *e.TargetID != b.ID {
		t.Fatalf("lost edge data: %+v", e)
	}
	if events[0].GraphIncarnation == nil || events[0].Timestamp <= 9007199254740991 {
		t.Fatalf("lost exact coordinates: %+v", events[0])
	}
	if _, err := db.ChangesAfter(cursor, 1, 4096); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed read: %v", err)
	}
}

func TestCDCStructuredErrors(t *testing.T) {
	db := cdcDB(t)
	if _, err := db.CreateNode([]string{"N"}, nil); err != nil {
		t.Fatal(err)
	}
	for _, cursor := range [][]byte{{}, {1}, make([]byte, 97)} {
		_, err := db.ChangesAfter(cursor, 1, 4096)
		cdcCode(t, err, "GRAFEO-S004")
	}
	for _, limits := range [][2]int{{0, 4096}, {-1, 4096}, {1, 0}, {1, -1}} {
		_, err := db.ChangesAfter(nil, limits[0], limits[1])
		cdcCode(t, err, "GRAFEO-V001")
	}
	_, err := db.ChangesAfter(nil, 1, 1)
	cdcCode(t, err, "GRAFEO-S001")
	page, err := db.ChangesAfter(nil, 1, 4096)
	if err != nil {
		t.Fatal(err)
	}
	other := cdcDB(t)
	_, err = other.ChangesAfter(page.Next, 1, 4096)
	cdcCode(t, err, "GRAFEO-S005")
}

func TestCDCCursorResumesAcrossTwoDirectoryReopens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store")
	db, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := db.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := db.SetCDCEnabled(true); err != nil {
		t.Fatal(err)
	}
	var ids []uint64
	for i := 0; i < 3; i++ {
		node, err := db.CreateNode([]string{"N"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, node.ID)
	}
	var cursor []byte
	for index, id := range ids {
		if index > 0 {
			db, err = Open(path)
			if err != nil {
				t.Fatal(err)
			}
		}
		page, err := db.ChangesAfter(cursor, 1, 4096)
		if err != nil {
			t.Fatal(err)
		}
		if len(page.Events) != 1 || page.Events[0].EntityID != id {
			t.Fatalf("reopened page: %+v", page)
		}
		cursor = page.Next
		if err := db.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCDCExactJSONCoordinates(t *testing.T) {
	// Exercise Go's decoder above the precision of a float64 and signed int64.
	input := `[{"entity_id":"18446744073709551614","epoch":"9007199254740993","timestamp":"18446744073709551613","graph_incarnation":"18446744073709551612","src_id":"18446744073709551611","dst_id":"18446744073709551610"}]`
	var events []ChangeEvent
	decoder := json.NewDecoder(strings.NewReader(input))
	decoder.UseNumber()
	if err := decoder.Decode(&events); err != nil {
		t.Fatal(err)
	}
	e := events[0]
	if e.EntityID != ^uint64(0)-1 || e.Epoch != 9007199254740993 || e.Timestamp != ^uint64(0)-2 || e.GraphIncarnation == nil || *e.GraphIncarnation != ^uint64(0)-3 || e.SourceID == nil || *e.SourceID != ^uint64(0)-4 || e.TargetID == nil || *e.TargetID != ^uint64(0)-5 {
		t.Fatalf("narrowed coordinates: %+v", e)
	}
}

func TestCDCEvictedCursorKeepsNativeCode(t *testing.T) {
	path := os.Getenv("GRAFEO_CDC_EVICTED_FIXTURE")
	if path == "" {
		t.Skip("cross-language fixture must be generated by the C retention control")
	}
	cursorPath := strings.TrimSuffix(path, filepath.Ext(path)) + ".cursor"
	cursor, err := os.ReadFile(cursorPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(cursor) != 97 {
		t.Fatalf("invalid fixture cursor length: %d", len(cursor))
	}
	// Opening/closing can rewrite container metadata. Preserve the shared
	// retained-cut witness for every other binding by opening a private copy.
	cut, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ownedPath := filepath.Join(t.TempDir(), "retained.grafeo")
	if err := os.WriteFile(ownedPath, cut, 0600); err != nil {
		t.Fatal(err)
	}
	db, err := Open(ownedPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	_, err = db.ChangesAfter(cursor, 1, 4096)
	cdcCode(t, err, "GRAFEO-S006")
	fresh, err := db.ChangesAfter(nil, 1, 4096)
	if err != nil || len(fresh.Events) != 1 {
		t.Fatalf("retained cut: %+v %v", fresh, err)
	}
}
