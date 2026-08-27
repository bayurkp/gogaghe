// internal/store/ngram_test.go
package store_test

import (
	"testing"

	"github.com/bayurkp/gogaghe/internal/store"
)

func TestExtractNgrams(t *testing.T) {
	tests := []struct {
		input    string
		n        int
		expected []string
	}{
		{
			input:    "PostgreSQL",
			n:        3,
			expected: []string{"pos", "ost", "stg", "tgr", "gre", "res", "esq", "sql"},
		},
		{
			input:    "go",
			n:        3,
			expected: []string{"go"}, // shorter than n
		},
		{
			input:    "Redis_16",
			n:        3,
			expected: []string{"red", "edi", "dis", "16"},
		},
	}

	for _, tc := range tests {
		got := store.ExtractNgrams(tc.input, tc.n)
		if len(got) != len(tc.expected) {
			t.Fatalf("ExtractNgrams(%q, %d) len = %d, want %d: %v vs %v", tc.input, tc.n, len(got), len(tc.expected), got, tc.expected)
		}
		for i := range got {
			if got[i] != tc.expected[i] {
				t.Errorf("ExtractNgrams(%q, %d)[%d] = %q, want %q", tc.input, tc.n, i, got[i], tc.expected[i])
			}
		}
	}
}

func TestNgramIndex_Search_TypoTolerance(t *testing.T) {
	idx := store.NewNgramIndex(3)
	items := map[string]store.Item{
		"doc-pg":     {Value: []byte("PostgreSQL database setup")},
		"doc-redis":  {Value: []byte("Redis cache memory")},
		"doc-k8s":    {Value: []byte("kubernetes cluster orchestration")},
		"doc-iphone": {Value: []byte("Apple iPhone 15 Pro Max")},
	}
	idx.Rebuild(items)

	// 1. Typo query: "Postgres" matching "PostgreSQL"
	pgResults := idx.Search("Postgres", 5)
	if len(pgResults) == 0 || pgResults[0].Key != "doc-pg" {
		t.Fatalf("expected doc-pg as top result for 'Postgres', got: %v", pgResults)
	}

	// 2. Typo query: "kubernets" matching "kubernetes"
	k8sResults := idx.Search("kubernets", 5)
	if len(k8sResults) == 0 || k8sResults[0].Key != "doc-k8s" {
		t.Fatalf("expected doc-k8s as top result for 'kubernets', got: %v", k8sResults)
	}

	// 3. Typo query: "iphon" matching "iPhone"
	iphoneResults := idx.Search("iphon", 5)
	if len(iphoneResults) == 0 || iphoneResults[0].Key != "doc-iphone" {
		t.Fatalf("expected doc-iphone as top result for 'iphon', got: %v", iphoneResults)
	}

	// 4. Non-matching query
	emptyResults := idx.Search("zzzzxxxx9999", 5)
	if len(emptyResults) != 0 {
		t.Fatalf("expected 0 results for non-matching query, got: %v", emptyResults)
	}
}

func TestNgramIndex_Incremental(t *testing.T) {
	idx := store.NewNgramIndex(3)

	idx.IndexDocument("doc1", []byte("Postgres cluster configuration"))
	idx.IndexDocument("doc2", []byte("Redis cache memory"))

	res := idx.Search("Postgres", 2)
	if len(res) == 0 || res[0].Key != "doc1" {
		t.Fatalf("expected doc1 for Postgres, got %v", res)
	}

	// Update doc1 with unrelated content
	idx.IndexDocument("doc1", []byte("Memcached fast lookup"))
	resAfterUpdate := idx.Search("Postgres", 2)
	if len(resAfterUpdate) != 0 {
		t.Fatalf("expected 0 results for Postgres after update, got %v", resAfterUpdate)
	}

	// Remove doc2
	idx.RemoveDocument("doc2")
	resAfterRemove := idx.Search("Redis", 2)
	if len(resAfterRemove) != 0 {
		t.Fatalf("expected 0 results for Redis after remove, got %v", resAfterRemove)
	}
}

func TestRRFMulti_ThreeStreams(t *testing.T) {
	// Surface (N-gram)
	ngram := []store.ScoredKey{
		{Key: "doc-typo", Score: 0.9},
		{Key: "doc-exact", Score: 0.8},
	}
	// Lexical (BM25)
	bm25 := []store.ScoredKey{
		{Key: "doc-exact", Score: 12.0},
		{Key: "doc-keyword", Score: 8.0},
	}
	// Semantic (Dense Vector)
	vec := []store.ScoredKey{
		{Key: "doc-exact", Score: 0.95},
		{Key: "doc-semantic", Score: 0.90},
	}

	results := store.RRFMulti([][]store.ScoredKey{ngram, bm25, vec}, 4, 60.0)
	if len(results) != 4 {
		t.Fatalf("expected 4 fused results, got %d", len(results))
	}

	// "doc-exact" appeared in all 3 lists (rank 2 in ngram, rank 1 in bm25, rank 1 in vec)
	// it should be clearly #1
	if results[0].Key != "doc-exact" {
		t.Errorf("expected doc-exact as #1 rank, got %s", results[0].Key)
	}
}

func TestNgramIndex_Metrics_Dice_Jaccard_Overlap(t *testing.T) {
	idx := store.NewNgramIndex(3)
	items := map[string]store.Item{
		"doc1": {Value: []byte("iPhone 15 Pro Max")},
		"doc2": {Value: []byte("Samsung Galaxy Ultra")},
	}
	idx.Rebuild(items)

	// Dice (default)
	diceRes := idx.SearchWithMetric("iPhone", 2, store.MetricDice)
	if len(diceRes) == 0 || diceRes[0].Key != "doc1" {
		t.Fatalf("expected doc1 for Dice metric, got %v", diceRes)
	}

	// Jaccard
	jaccardRes := idx.SearchWithMetric("iPhone", 2, store.MetricJaccard)
	if len(jaccardRes) == 0 || jaccardRes[0].Key != "doc1" {
		t.Fatalf("expected doc1 for Jaccard metric, got %v", jaccardRes)
	}

	// Overlap
	overlapRes := idx.SearchWithMetric("iPhone", 2, store.MetricOverlap)
	if len(overlapRes) == 0 || overlapRes[0].Key != "doc1" {
		t.Fatalf("expected doc1 for Overlap metric, got %v", overlapRes)
	}
}
