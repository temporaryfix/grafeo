package grafeo

/*
#include "grafeo.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// CreateIndex commits one catalog owner. Empty Graph selects root; graph
// components are UTF-8 byte spans and are never split on punctuation.
func (db *Database) CreateIndex(request CreateIndexRequest) (IndexID, error) {
	if err := db.acquire(); err != nil {
		return 0, err
	}
	defer db.release()
	native := (*C.GrafeoIndexRequest)(C.calloc(1, C.size_t(unsafe.Sizeof(C.GrafeoIndexRequest{}))))
	if native == nil {
		return 0, errors.New("index request allocation failed")
	}
	defer C.free(unsafe.Pointer(native))
	var allocations []unsafe.Pointer
	defer func() {
		for _, p := range allocations {
			C.free(p)
		}
	}()
	span := func(value string) (C.GrafeoUtf8, error) {
		if len(value) == 0 {
			return C.GrafeoUtf8{}, nil
		}
		p := C.malloc(C.size_t(len(value)))
		if p == nil {
			return C.GrafeoUtf8{}, errors.New("index UTF-8 allocation failed")
		}
		allocations = append(allocations, p)
		copy(unsafe.Slice((*byte)(p), len(value)), value)
		return C.GrafeoUtf8{data: (*C.uint8_t)(p), len: C.size_t(len(value))}, nil
	}
	native.kind = C.uint32_t(request.Kind)
	var err error
	native.property, err = span(request.Property)
	if err != nil {
		return 0, err
	}
	if len(request.Graph) != 0 {
		p := C.calloc(C.size_t(len(request.Graph)), C.size_t(unsafe.Sizeof(C.GrafeoUtf8{})))
		if p == nil {
			return 0, errors.New("index graph path allocation failed")
		}
		allocations = append(allocations, p)
		native.graph = (*C.GrafeoUtf8)(p)
		native.graph_count = C.size_t(len(request.Graph))
		components := unsafe.Slice(native.graph, len(request.Graph))
		for i, component := range request.Graph {
			components[i], err = span(component)
			if err != nil {
				return 0, err
			}
		}
	}
	for _, option := range []struct {
		value  *string
		bit    C.uint32_t
		target *C.GrafeoUtf8
	}{
		{request.Name, 1, &native.name}, {request.Label, 2, &native.label},
		{request.Metric, 8, &native.metric}, {request.Quantization, 64, &native.quantization},
	} {
		if option.value != nil {
			native.options |= option.bit
			*option.target, err = span(*option.value)
			if err != nil {
				return 0, err
			}
		}
	}
	if request.Dimensions != nil {
		native.options |= 4
		native.dimensions = C.size_t(*request.Dimensions)
	}
	if request.M != nil {
		native.options |= 16
		native.m = C.size_t(*request.M)
	}
	if request.EfConstruction != nil {
		native.options |= 32
		native.ef_construction = C.size_t(*request.EfConstruction)
	}
	if request.MinTokenLength != nil {
		length := C.size_t(*request.MinTokenLength)
		if uint(length) != *request.MinTokenLength {
			return 0, errors.New("min token length exceeds native size_t")
		}
		native.options |= 128
		native.min_token_length = length
	}
	var owner C.uint32_t
	err = lockAndCheckStatus(func() C.GrafeoStatus { return C.grafeo_create_index(db.handle, native, &owner) })
	return IndexID(owner), err
}

// DropIndex removes this exact owner; absence is false, failures are errors.
func (db *Database) DropIndex(owner IndexID) (bool, error) {
	if err := db.acquire(); err != nil {
		return false, err
	}
	defer db.release()
	var dropped C.int32_t
	err := lockAndCheckStatus(func() C.GrafeoStatus { return C.grafeo_drop_index(db.handle, C.uint32_t(owner), &dropped) })
	return dropped != 0, err
}

// RebuildIndex retains the owner's resolved configuration; absence is an error.
func (db *Database) RebuildIndex(owner IndexID) error {
	if err := db.acquire(); err != nil {
		return err
	}
	defer db.release()
	return lockAndCheckStatus(func() C.GrafeoStatus { return C.grafeo_rebuild_index(db.handle, C.uint32_t(owner)) })
}

// HasPropertyIndex checks whether a property index exists.
func (db *Database) HasPropertyIndex(property string) bool {
	if err := db.acquire(); err != nil {
		return false
	}
	defer db.release()
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))
	runtime.LockOSThread()
	result := int(C.grafeo_has_property_index(db.handle, cProp)) == 1
	runtime.UnlockOSThread()
	return result
}

// FindNodesByProperty finds nodes with a matching property value.
func (db *Database) FindNodesByProperty(property string, value any) ([]uint64, error) {
	if err := db.acquire(); err != nil {
		return nil, err
	}
	defer db.release()
	cProp := C.CString(property)
	defer C.free(unsafe.Pointer(cProp))

	valueJSON, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	cValue := C.CString(string(valueJSON))
	defer C.free(unsafe.Pointer(cValue))

	var outIDs *C.uint64_t
	var outCount C.size_t

	runtime.LockOSThread()
	status := C.grafeo_find_nodes_by_property(db.handle, cProp, cValue, &outIDs, &outCount)
	if status != C.GRAFEO_OK {
		err := statusToError(status)
		runtime.UnlockOSThread()
		return nil, err
	}
	runtime.UnlockOSThread()

	count := int(outCount)
	if count == 0 {
		return nil, nil
	}
	defer C.grafeo_free_node_ids(outIDs, outCount)

	ids := make([]uint64, count)
	raw := unsafe.Slice((*uint64)(unsafe.Pointer(outIDs)), count)
	copy(ids, raw)
	return ids, nil
}
