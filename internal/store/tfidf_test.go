// internal/store/tfidf_test.go
package store_test

import (
	"testing"

	"github.com/bayurkp/gogaghe/internal/store"
)

func TestTFIDFIndex_Search(t *testing.T) {
	idx := store.NewTFIDFIndex()

	docs := map[string]store.Item{
		"doc1": {Value: []byte("golang fast in-memory cache system")},
		"doc2": {Value: []byte("python machine learning deep learning neural network")},
		"doc3": {Value: []byte("golang concurrency goroutines channels")},
	}

	idx.Rebuild(docs)

	// Search for golang
	results := idx.Search("golang cache", 5)
	if len(results) == 0 {
		t.Fatalf("expected results for query 'golang cache', got 0")
	}

	if results[0].Key != "doc1" {
		t.Errorf("expected top result doc1, got %s (score %f)", results[0].Key, results[0].Score)
	}

	// Incremental removal
	idx.RemoveDocument("doc1")
	resultsAfter := idx.Search("golang cache", 5)
	for _, r := range resultsAfter {
		if r.Key == "doc1" {
			t.Errorf("doc1 should have been removed from index")
		}
	}
}
