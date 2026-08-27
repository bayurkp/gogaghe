// internal/store/lsa_test.go
package store_test

import (
	"testing"

	"github.com/bayurkp/gogaghe/internal/store"
)

func TestLSAIndex_Search(t *testing.T) {
	idx := store.NewLSAIndexWithDim(8)

	docs := map[string]store.Item{
		"doc1": {Value: []byte("artificial intelligence deep learning neural network model")},
		"doc2": {Value: []byte("machine learning statistical algorithm data science")},
		"doc3": {Value: []byte("golang web server http microservice backend")},
		"doc4": {Value: []byte("cloud native kubernetes docker container orchestration")},
	}

	idx.Rebuild(docs)

	// Search for AI terms
	results := idx.Search("deep learning ai neural", 5)
	if len(results) == 0 {
		t.Fatalf("expected results for query 'deep learning ai neural', got 0")
	}

	if results[0].Key != "doc1" {
		t.Logf("top result: %s (score: %f)", results[0].Key, results[0].Score)
	}

	// Empty query
	emptyResults := idx.Search("", 5)
	if len(emptyResults) != 0 {
		t.Errorf("expected 0 results for empty query, got %d", len(emptyResults))
	}
}
