// internal/store/engine_test.go
package store_test

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/bayurkp/gogaghe/internal/store"
)

func newTestEngine(t *testing.T) *store.Engine {
	t.Helper()
	e := store.NewEngine(64*1024*1024, 100*time.Millisecond)
	t.Cleanup(e.Stop)
	return e
}

func TestSetAndGet(t *testing.T) {
	e := newTestEngine(t)
	item := store.Item{Value: []byte("hello"), CostMs: 10}
	if err := e.Set("k1", item); err != nil {
		t.Fatalf("Set error: %v", err)
	}
	got, ok := e.Get("k1")
	if !ok {
		t.Fatal("Get: key not found")
	}
	if string(got.Value) != "hello" {
		t.Errorf("Value = %q, want %q", got.Value, "hello")
	}
	if got.AccessCount != 1 {
		t.Errorf("AccessCount = %d, want 1", got.AccessCount)
	}
	if e.Len() != 1 {
		t.Errorf("Len = %d, want 1", e.Len())
	}
	if e.MemoryUsageBytes() <= 0 {
		t.Errorf("MemoryUsageBytes = %d, want > 0", e.MemoryUsageBytes())
	}
}

func TestDelete(t *testing.T) {
	e := newTestEngine(t)
	_ = e.Set("k2", store.Item{Value: []byte("bye")})
	if !e.Delete("k2") {
		t.Error("Delete should return true for existing key")
	}
	if _, ok := e.Get("k2"); ok {
		t.Error("key should not exist after Delete")
	}
	if e.Delete("k2") {
		t.Error("Delete should return false for missing key")
	}
	if e.Len() != 0 {
		t.Errorf("Len = %d, want 0", e.Len())
	}
	if e.MemoryUsageBytes() != 0 {
		t.Errorf("MemoryUsageBytes = %d, want 0", e.MemoryUsageBytes())
	}
}

func TestTTLExpiry(t *testing.T) {
	e := newTestEngine(t)
	item := store.Item{
		Value:     []byte("expire me"),
		ExpiresAt: time.Now().Add(50 * time.Millisecond),
	}
	_ = e.Set("ttlkey", item)
	time.Sleep(200 * time.Millisecond)
	if _, ok := e.Get("ttlkey"); ok {
		t.Error("expected key to be expired on Get")
	}
}

func TestMemoryLimit(t *testing.T) {
	// Small engine: 100 bytes
	e := store.NewEngine(100, 1*time.Second)
	t.Cleanup(e.Stop)

	bigValue := make([]byte, 200)
	err := e.Set("big", store.Item{Value: bigValue})
	if err == nil {
		t.Error("expected error when setting item exceeding memory limit, got nil")
	}
}

func TestConcurrentAccess(t *testing.T) {
	e := newTestEngine(t)
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", i)
			_ = e.Set(key, store.Item{Value: []byte("v"), CostMs: int64(i)})
			_, _ = e.Get(key)
			_ = e.Len()
			_ = e.MemoryUsageBytes()
			_ = e.Items()
			if i%2 == 0 {
				_ = e.Delete(key)
			}
		}(i)
	}
	wg.Wait()
}

func TestBM25Search(t *testing.T) {
	idx := store.NewBM25Index()
	items := map[string]store.Item{
		"doc1": {Value: []byte("the quick brown fox jumps over the lazy dog")},
		"doc2": {Value: []byte("quick brown fox")},
		"doc3": {Value: []byte("lazy dog sleeps all day")},
	}
	idx.Rebuild(items)

	results := idx.Search("quick fox", 3)
	if len(results) == 0 {
		t.Fatal("expected results, got none")
	}
	// doc1 and doc2 should score highest for "quick fox"
	top := results[0].Key
	if top != "doc1" && top != "doc2" {
		t.Errorf("unexpected top result key: %s", top)
	}

	// Search for something only in doc3
	res3 := idx.Search("sleeps", 1)
	if len(res3) != 1 || res3[0].Key != "doc3" {
		t.Errorf("expected doc3, got %v", res3)
	}

	// Search for non-existent token
	resEmpty := idx.Search("nonexistentword123", 5)
	if len(resEmpty) != 0 {
		t.Errorf("expected 0 results, got %d", len(resEmpty))
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"identical", []float32{1, 0}, []float32{1, 0}, 1.0},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0.0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1.0},
		{"zero vector", []float32{0, 0}, []float32{1, 1}, 0.0},
		{"diff length", []float32{1, 0}, []float32{1, 0, 0}, 0.0},
		{"empty", []float32{}, []float32{}, 0.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := store.CosineSimilarity(tc.a, tc.b)
			if math.Abs(got-tc.want) > 1e-6 {
				t.Errorf("CosineSimilarity(%v, %v) = %f, want %f", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestVectorSearch(t *testing.T) {
	items := map[string]store.Item{
		"a": {Vector: []float32{1, 0, 0}},
		"b": {Vector: []float32{0, 1, 0}},
		"c": {Vector: []float32{0.9, 0.1, 0}}, // closest to [1, 0, 0]
		"d": {Vector: []float32{1, 0}},       // dimension mismatch -> skipped
		"e": {Vector: nil},                  // no vector -> skipped
	}
	query := []float32{1, 0, 0}
	results := store.VectorSearch(query, items, 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Key != "a" {
		t.Errorf("expected top result 'a', got %s", results[0].Key)
	}
	if results[1].Key != "c" {
		t.Errorf("expected second result 'c', got %s", results[1].Key)
	}
}
