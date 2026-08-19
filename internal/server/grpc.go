// internal/server/grpc.go
package server

import (
	"context"
	"runtime"
	"time"

	"github.com/bayurkp/gogaghe/internal/embedder"
	"github.com/bayurkp/gogaghe/internal/store"
	gogaghev1 "github.com/bayurkp/gogaghe/pkg/gogaghe/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GogagheServer implements the GogagheServiceServer gRPC interface.
type GogagheServer struct {
	gogaghev1.UnimplementedGogagheServiceServer
	engine   *store.Engine
	bm25     *store.BM25Index
	metrics  *Metrics
	embedder *embedder.Client
}

// NewGogagheServer creates a GogagheServer wired to the given subsystems.
func NewGogagheServer(
	engine *store.Engine,
	bm25 *store.BM25Index,
	metrics *Metrics,
	emb *embedder.Client,
) *GogagheServer {
	return &GogagheServer{
		engine:   engine,
		bm25:     bm25,
		metrics:  metrics,
		embedder: emb,
	}
}

func (s *GogagheServer) Set(ctx context.Context, req *gogaghev1.SetRequest) (*gogaghev1.SetResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("set"))
	defer timer()

	var expiresAt time.Time
	if req.TtlMs > 0 {
		expiresAt = time.Now().Add(time.Duration(req.TtlMs) * time.Millisecond)
	}

	item := store.Item{
		Value:     req.Value,
		CostMs:    req.CostMs,
		Vector:    req.Vector,
		ExpiresAt: expiresAt,
	}

	if err := s.engine.Set(req.Key, item); err != nil {
		return nil, status.Errorf(codes.ResourceExhausted, "set failed: %v", err)
	}

	// Incrementally update BM25 index for O(tokens) write complexity
	s.bm25.IndexDocument(req.Key, req.Value)

	// Async embedding if requested and embedder is configured.
	if req.AutoEmbed && s.embedder != nil {
		s.embedder.Enqueue(embedder.EmbedRequest{
			Key:  req.Key,
			Text: string(req.Value),
		})
	}

	s.metrics.ItemsCount.Set(float64(s.engine.Len()))
	s.metrics.MemoryUsageBytes.Set(float64(s.engine.MemoryUsageBytes()))
	s.metrics.GoroutinesActive.Set(float64(runtime.NumGoroutine()))

	return &gogaghev1.SetResponse{Success: true}, nil
}

func (s *GogagheServer) Get(ctx context.Context, req *gogaghev1.GetRequest) (*gogaghev1.GetResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("get"))
	defer timer()

	item, ok := s.engine.Get(req.Key)
	if !ok {
		s.metrics.CacheMisses.Inc()
		return &gogaghev1.GetResponse{Found: false}, nil
	}

	s.metrics.CacheHits.Inc()
	s.metrics.GoroutinesActive.Set(float64(runtime.NumGoroutine()))

	return &gogaghev1.GetResponse{
		Found:       true,
		Value:       item.Value,
		Vector:      item.Vector,
		AccessCount: item.AccessCount,
	}, nil
}

func (s *GogagheServer) Delete(ctx context.Context, req *gogaghev1.DeleteRequest) (*gogaghev1.DeleteResponse, error) {
	deleted := s.engine.Delete(req.Key)
	if deleted {
		s.bm25.RemoveDocument(req.Key)
	}
	s.metrics.ItemsCount.Set(float64(s.engine.Len()))
	s.metrics.MemoryUsageBytes.Set(float64(s.engine.MemoryUsageBytes()))
	return &gogaghev1.DeleteResponse{Deleted: deleted}, nil
}

func (s *GogagheServer) VectorSearch(ctx context.Context, req *gogaghev1.VectorSearchRequest) (*gogaghev1.VectorSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("vector_search"))
	defer timer()

	results := store.VectorSearch(req.QueryVector, s.engine.Items(), int(req.TopK))
	return &gogaghev1.VectorSearchResponse{Results: toProtoResults(results, s.engine)}, nil
}

func (s *GogagheServer) HybridSearch(ctx context.Context, req *gogaghev1.HybridSearchRequest) (*gogaghev1.HybridSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("hybrid_search"))
	defer timer()

	items := s.engine.Items()
	topK := int(req.TopK)
	k := float64(req.RrfK)
	if k <= 0 {
		k = 60.0
	}

	bm25Results := s.bm25.Search(req.Query, topK*2)
	vecResults := store.VectorSearch(req.QueryVector, items, topK*2)
	fused := store.RRF(bm25Results, vecResults, topK, k)

	return &gogaghev1.HybridSearchResponse{Results: toProtoResults(fused, s.engine)}, nil
}

// toProtoResults converts []ScoredKey to []*gogaghev1.SearchResult, enriching
// with Value from the engine snapshot.
func toProtoResults(scored []store.ScoredKey, e *store.Engine) []*gogaghev1.SearchResult {
	out := make([]*gogaghev1.SearchResult, 0, len(scored))
	for _, s := range scored {
		item, ok := e.Get(s.Key)
		var val []byte
		if ok {
			val = item.Value
		}
		out = append(out, &gogaghev1.SearchResult{
			Key:   s.Key,
			Value: val,
			Score: float32(s.Score),
		})
	}
	return out
}

// prometheusTimer returns a closure observing elapsed duration on the given Observer.
func prometheusTimer(obs interface{ Observe(float64) }) func() {
	start := time.Now()
	return func() { obs.Observe(time.Since(start).Seconds()) }
}
