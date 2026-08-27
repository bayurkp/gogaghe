// internal/server/grpc_test.go
package server_test

import (
	"context"
	"testing"
	"time"

	"github.com/bayurkp/gogaghe/internal/server"
	"github.com/bayurkp/gogaghe/internal/store"
	gogaghev1 "github.com/bayurkp/gogaghe/pkg/gogaghe/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func newTestServer(t *testing.T, maxMem int64) (*server.GogagheServer, *store.Engine) {
	t.Helper()
	e := store.NewEngine(maxMem, 100*time.Millisecond)
	t.Cleanup(e.Stop)
	bm25 := store.NewBM25Index()
	ngram := store.NewNgramIndex(3)
	metrics := server.NewMetrics()
	srv := server.NewGogagheServer(e, bm25, ngram, metrics, nil)
	return srv, e
}

func TestGogagheServer_SetAndGet(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	// Set
	setResp, err := srv.Set(ctx, &gogaghev1.SetRequest{
		Key:    "test-key",
		Value:  []byte("test value content"),
		CostMs: 50,
		Vector: []float32{0.1, 0.2, 0.3},
	})
	if err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	if !setResp.Success {
		t.Error("expected SetResponse.Success to be true")
	}

	// Get existing
	getResp, err := srv.Get(ctx, &gogaghev1.GetRequest{Key: "test-key"})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if !getResp.Found {
		t.Fatal("expected Found to be true")
	}
	if string(getResp.Value) != "test value content" {
		t.Errorf("got value %q, want %q", getResp.Value, "test value content")
	}
	if len(getResp.Vector) != 3 {
		t.Errorf("got vector len %d, want 3", len(getResp.Vector))
	}

	// Get non-existent
	getMiss, err := srv.Get(ctx, &gogaghev1.GetRequest{Key: "missing-key"})
	if err != nil {
		t.Fatalf("Get miss failed: %v", err)
	}
	if getMiss.Found {
		t.Error("expected Found to be false for missing key")
	}
}

func TestGogagheServer_Delete(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{Key: "k1", Value: []byte("val1")})

	delResp, err := srv.Delete(ctx, &gogaghev1.DeleteRequest{Key: "k1"})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if !delResp.Deleted {
		t.Error("expected Deleted = true")
	}

	delResp2, err := srv.Delete(ctx, &gogaghev1.DeleteRequest{Key: "k1"})
	if err != nil {
		t.Fatalf("Delete miss failed: %v", err)
	}
	if delResp2.Deleted {
		t.Error("expected Deleted = false for non-existent key")
	}
}

func TestGogagheServer_VectorSearch(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{Key: "doc1", Value: []byte("v1"), Vector: []float32{1, 0, 0}})
	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{Key: "doc2", Value: []byte("v2"), Vector: []float32{0, 1, 0}})

	res, err := srv.VectorSearch(ctx, &gogaghev1.VectorSearchRequest{
		QueryVector: []float32{1, 0, 0},
		TopK:        1,
	})
	if err != nil {
		t.Fatalf("VectorSearch failed: %v", err)
	}
	if len(res.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res.Results))
	}
	if res.Results[0].Key != "doc1" {
		t.Errorf("expected doc1, got %s", res.Results[0].Key)
	}
}

func TestGogagheServer_HybridSearch(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:    "doc1",
		Value:  []byte("golang in-memory hybrid store"),
		Vector: []float32{1, 0, 0},
	})
	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:    "doc2",
		Value:  []byte("python postgres transactional database"),
		Vector: []float32{0, 1, 0},
	})

	res, err := srv.HybridSearch(ctx, &gogaghev1.HybridSearchRequest{
		Query:       "golang hybrid",
		QueryVector: []float32{0.9, 0.1, 0},
		TopK:        2,
	})
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	if len(res.Results) == 0 {
		t.Fatal("expected at least 1 hybrid search result")
	}
	if res.Results[0].Key != "doc1" {
		t.Errorf("expected doc1 as top result, got %s", res.Results[0].Key)
	}
}

func TestGogagheServer_SurfaceSearch(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:   "doc-iphone",
		Value: []byte("Apple iPhone 15 Pro Max"),
	})
	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:   "doc-samsung",
		Value: []byte("Samsung Galaxy S24 Ultra"),
	})

	// Search with typo query: "iphon"
	res, err := srv.SurfaceSearch(ctx, &gogaghev1.SurfaceSearchRequest{
		Query: "iphon",
		TopK:  2,
	})
	if err != nil {
		t.Fatalf("SurfaceSearch failed: %v", err)
	}
	if len(res.Results) == 0 || res.Results[0].Key != "doc-iphone" {
		t.Fatalf("expected doc-iphone as top surface search result, got: %v", res.Results)
	}
}

func TestGogagheServer_LexicalSearch(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:   "doc1",
		Value: []byte("distributed database consensus algorithm raft"),
	})
	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:   "doc2",
		Value: []byte("in-memory cache eviction policy lru"),
	})

	res, err := srv.LexicalSearch(ctx, &gogaghev1.LexicalSearchRequest{
		Query: "consensus raft",
		TopK:  1,
	})
	if err != nil {
		t.Fatalf("LexicalSearch failed: %v", err)
	}
	if len(res.Results) == 0 || res.Results[0].Key != "doc1" {
		t.Fatalf("expected doc1 as top lexical search result, got: %v", res.Results)
	}
}

func TestGogagheServer_HybridSearch_StrategySelection(t *testing.T) {
	srv, _ := newTestServer(t, 1024*1024)
	ctx := context.Background()

	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:    "doc-pg",
		Value:  []byte("PostgreSQL database setup"),
		Vector: []float32{1, 0, 0},
	})
	_, _ = srv.Set(ctx, &gogaghev1.SetRequest{
		Key:    "doc-redis",
		Value:  []byte("Redis in-memory caching"),
		Vector: []float32{0, 1, 0},
	})

	// 1. Explicitly select SURFACE + SEMANTIC (Skip Lexical)
	res1, err := srv.HybridSearch(ctx, &gogaghev1.HybridSearchRequest{
		Query:       "Postgres", // typo / surface match for PostgreSQL
		QueryVector: []float32{0.9, 0.1, 0},
		TopK:        2,
		Strategies: []gogaghev1.SearchStrategy{
			gogaghev1.SearchStrategy_SEARCH_STRATEGY_SURFACE,
			gogaghev1.SearchStrategy_SEARCH_STRATEGY_SEMANTIC,
		},
	})
	if err != nil {
		t.Fatalf("HybridSearch with strategies failed: %v", err)
	}
	if len(res1.Results) == 0 || res1.Results[0].Key != "doc-pg" {
		t.Errorf("expected doc-pg as top result for Surface+Semantic, got: %v", res1.Results)
	}

	// 2. Explicitly select only LEXICAL (BM25)
	res2, err := srv.HybridSearch(ctx, &gogaghev1.HybridSearchRequest{
		Query:       "Postgres", // BM25 exact match won't match "PostgreSQL"
		QueryVector: []float32{0.9, 0.1, 0}, // provided but ignored because SEMANTIC is not in strategies
		TopK:        2,
		Strategies: []gogaghev1.SearchStrategy{
			gogaghev1.SearchStrategy_SEARCH_STRATEGY_LEXICAL,
		},
	})
	if err != nil {
		t.Fatalf("HybridSearch with lexical only failed: %v", err)
	}
	// "Postgres" token doesn't exist in BM25 ("postgresql" does) -> 0 results
	if len(res2.Results) != 0 {
		t.Errorf("expected 0 results for exact lexical search on 'Postgres', got: %v", res2.Results)
	}
}

func TestGogagheServer_ResourceExhausted(t *testing.T) {
	srv, _ := newTestServer(t, 50) // only 50 bytes
	ctx := context.Background()

	_, err := srv.Set(ctx, &gogaghev1.SetRequest{
		Key:   "large-key",
		Value: make([]byte, 200),
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.ResourceExhausted {
		t.Errorf("expected codes.ResourceExhausted, got %v", err)
	}
}
