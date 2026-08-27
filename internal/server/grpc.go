// internal/server/grpc.go
package server

import (
	"context"
	"log/slog"
	"runtime"
	"time"

	"github.com/bayurkp/gogaghe/internal/config"
	"github.com/bayurkp/gogaghe/internal/embedder"
	"github.com/bayurkp/gogaghe/internal/store"
	gogaghev1 "github.com/bayurkp/gogaghe/pkg/gogaghe/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GogagheServer implements the GogagheServiceServer gRPC interface.
type GogagheServer struct {
	gogaghev1.UnimplementedGogagheServiceServer
	engine    *store.Engine
	bm25      *store.BM25Index
	ngram     *store.NgramIndex
	tfidf     *store.TFIDFIndex
	lsa       *store.LSAIndex
	metrics   *Metrics
	embedder  *embedder.Client
	searchCfg config.SearchConfig
}

// NewGogagheServer creates a GogagheServer wired to the given subsystems with default search config.
func NewGogagheServer(
	engine *store.Engine,
	bm25 *store.BM25Index,
	ngram *store.NgramIndex,
	tfidf *store.TFIDFIndex,
	lsa *store.LSAIndex,
	metrics *Metrics,
	emb *embedder.Client,
) *GogagheServer {
	return NewGogagheServerWithConfig(engine, bm25, ngram, tfidf, lsa, metrics, emb, config.DefaultConfig().Search)
}

// NewGogagheServerWithConfig creates a GogagheServer wired to the given subsystems and custom search tuning.
func NewGogagheServerWithConfig(
	engine *store.Engine,
	bm25 *store.BM25Index,
	ngram *store.NgramIndex,
	tfidf *store.TFIDFIndex,
	lsa *store.LSAIndex,
	metrics *Metrics,
	emb *embedder.Client,
	searchCfg config.SearchConfig,
) *GogagheServer {
	if searchCfg.Hybrid.DefaultRRFK <= 0 {
		searchCfg.Hybrid.DefaultRRFK = 60.0
	}
	return &GogagheServer{
		engine:    engine,
		bm25:      bm25,
		ngram:     ngram,
		tfidf:     tfidf,
		lsa:       lsa,
		metrics:   metrics,
		embedder:  emb,
		searchCfg: searchCfg,
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

	// Incrementally update search indexes
	if s.bm25 != nil {
		s.bm25.IndexDocument(req.Key, req.Value)
	}
	if s.ngram != nil {
		s.ngram.IndexDocument(req.Key, req.Value)
	}
	if s.tfidf != nil {
		s.tfidf.IndexDocument(req.Key, req.Value)
	}
	if s.lsa != nil {
		s.lsa.Rebuild(s.engine.Items())
	}

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
		if s.bm25 != nil {
			s.bm25.RemoveDocument(req.Key)
		}
		if s.ngram != nil {
			s.ngram.RemoveDocument(req.Key)
		}
		if s.tfidf != nil {
			s.tfidf.RemoveDocument(req.Key)
		}
		if s.lsa != nil {
			s.lsa.Rebuild(s.engine.Items())
		}
	}
	s.metrics.ItemsCount.Set(float64(s.engine.Len()))
	s.metrics.MemoryUsageBytes.Set(float64(s.engine.MemoryUsageBytes()))
	return &gogaghev1.DeleteResponse{Deleted: deleted}, nil
}

func (s *GogagheServer) SurfaceSearch(ctx context.Context, req *gogaghev1.SurfaceSearchRequest) (*gogaghev1.SurfaceSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("surface_search"))
	defer timer()

	var results []store.ScoredKey
	if s.ngram != nil && len(req.Query) > 0 {
		results = s.ngram.Search(req.Query, int(req.TopK))
	}
	return &gogaghev1.SurfaceSearchResponse{Results: toProtoResults(results, s.engine)}, nil
}

func (s *GogagheServer) Bm25Search(ctx context.Context, req *gogaghev1.Bm25SearchRequest) (*gogaghev1.Bm25SearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("bm25_search"))
	defer timer()

	var results []store.ScoredKey
	if s.bm25 != nil && len(req.Query) > 0 {
		results = s.bm25.Search(req.Query, int(req.TopK))
	}
	return &gogaghev1.Bm25SearchResponse{Results: toProtoResults(results, s.engine)}, nil
}

func (s *GogagheServer) LexicalSearch(ctx context.Context, req *gogaghev1.LexicalSearchRequest) (*gogaghev1.LexicalSearchResponse, error) {
	resp, err := s.Bm25Search(ctx, &gogaghev1.Bm25SearchRequest{
		Query: req.Query,
		TopK:  req.TopK,
	})
	if err != nil {
		return nil, err
	}
	return &gogaghev1.LexicalSearchResponse{Results: resp.Results}, nil
}

func (s *GogagheServer) TfidfSearch(ctx context.Context, req *gogaghev1.TfidfSearchRequest) (*gogaghev1.TfidfSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("tfidf_search"))
	defer timer()

	var results []store.ScoredKey
	if s.tfidf != nil && len(req.Query) > 0 {
		results = s.tfidf.Search(req.Query, int(req.TopK))
	}
	return &gogaghev1.TfidfSearchResponse{Results: toProtoResults(results, s.engine)}, nil
}

func (s *GogagheServer) LsaSearch(ctx context.Context, req *gogaghev1.LsaSearchRequest) (*gogaghev1.LsaSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("lsa_search"))
	defer timer()

	var results []store.ScoredKey
	if s.lsa != nil && len(req.Query) > 0 {
		results = s.lsa.Search(req.Query, int(req.TopK))
	}
	return &gogaghev1.LsaSearchResponse{Results: toProtoResults(results, s.engine)}, nil
}

func (s *GogagheServer) VectorSearch(ctx context.Context, req *gogaghev1.VectorSearchRequest) (*gogaghev1.VectorSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("vector_search"))
	defer timer()

	queryVec := req.QueryVector
	// Native query auto-embedding if query_vector is omitted but query text is provided
	if len(queryVec) == 0 && len(req.Query) > 0 && s.embedder != nil {
		vec, err := s.embedder.Embed(ctx, req.Query)
		if err != nil {
			slog.Warn("vector_search: auto-embed query failed", "query", req.Query, "err", err)
		} else {
			queryVec = vec
		}
	}

	results := store.VectorSearch(queryVec, s.engine.Items(), int(req.TopK))
	return &gogaghev1.VectorSearchResponse{Results: toProtoResults(results, s.engine)}, nil
}

func (s *GogagheServer) HybridSearch(ctx context.Context, req *gogaghev1.HybridSearchRequest) (*gogaghev1.HybridSearchResponse, error) {
	timer := prometheusTimer(s.metrics.OperationDuration.WithLabelValues("hybrid_search"))
	defer timer()

	items := s.engine.Items()
	topK := int(req.TopK)
	k := float64(req.RrfK)
	if k <= 0 {
		k = s.searchCfg.Hybrid.DefaultRRFK
	}
	if k <= 0 {
		k = 60.0
	}

	queryVec := req.QueryVector
	useSurface := false
	useBm25 := false
	useTfidf := false
	useLsa := false
	useDense := false

	if len(req.Strategies) == 0 {
		// Auto-detect based on provided fields and active engines
		useSurface = s.ngram != nil && len(req.Query) > 0
		useBm25 = s.bm25 != nil && len(req.Query) > 0
		useDense = len(queryVec) > 0 || (s.embedder != nil && len(req.Query) > 0)
	} else {
		for _, st := range req.Strategies {
			switch st {
			case gogaghev1.SearchStrategy_SEARCH_STRATEGY_SURFACE_NGRAM:
				useSurface = s.ngram != nil && len(req.Query) > 0
			case gogaghev1.SearchStrategy_SEARCH_STRATEGY_LEXICAL_BM25:
				useBm25 = s.bm25 != nil && len(req.Query) > 0
			case gogaghev1.SearchStrategy_SEARCH_STRATEGY_LEXICAL_TFIDF:
				useTfidf = s.tfidf != nil && len(req.Query) > 0
			case gogaghev1.SearchStrategy_SEARCH_STRATEGY_SEMANTIC_LSA:
				useLsa = s.lsa != nil && len(req.Query) > 0
			case gogaghev1.SearchStrategy_SEARCH_STRATEGY_SEMANTIC_DENSE:
				useDense = true
			case gogaghev1.SearchStrategy_SEARCH_STRATEGY_UNSPECIFIED:
				if s.ngram != nil && len(req.Query) > 0 {
					useSurface = true
				}
				if s.bm25 != nil && len(req.Query) > 0 {
					useBm25 = true
				}
				if len(queryVec) > 0 || (s.embedder != nil && len(req.Query) > 0) {
					useDense = true
				}
			}
		}
	}

	// Auto-embed query text for dense stream if query_vector is omitted
	if useDense && len(queryVec) == 0 && len(req.Query) > 0 && s.embedder != nil {
		vec, err := s.embedder.Embed(ctx, req.Query)
		if err != nil {
			slog.Warn("hybrid_search: auto-embed query failed", "query", req.Query, "err", err)
		} else {
			queryVec = vec
		}
	}

	var rankLists [][]store.ScoredKey

	if useSurface {
		ngramResults := s.ngram.Search(req.Query, topK*2)
		if len(ngramResults) > 0 {
			rankLists = append(rankLists, ngramResults)
		}
	}

	if useBm25 {
		bm25Results := s.bm25.Search(req.Query, topK*2)
		if len(bm25Results) > 0 {
			rankLists = append(rankLists, bm25Results)
		}
	}

	if useTfidf {
		tfidfResults := s.tfidf.Search(req.Query, topK*2)
		if len(tfidfResults) > 0 {
			rankLists = append(rankLists, tfidfResults)
		}
	}

	if useLsa {
		lsaResults := s.lsa.Search(req.Query, topK*2)
		if len(lsaResults) > 0 {
			rankLists = append(rankLists, lsaResults)
		}
	}

	if useDense && len(queryVec) > 0 {
		vecResults := store.VectorSearch(queryVec, items, topK*2)
		if len(vecResults) > 0 {
			rankLists = append(rankLists, vecResults)
		}
	}

	fused := store.RRFMulti(rankLists, topK, k)

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
