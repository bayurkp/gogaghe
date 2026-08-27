// internal/config/config_test.go
package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/bayurkp/gogaghe/internal/config"
)

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	yamlContent := `
server:
  grpc_port: 50052
  metrics_port: 2113
store:
  max_memory_bytes: 67108864
  ttl_check_interval_seconds: 15
  eviction_threshold_ratio: 0.85
search:
  surface:
    enabled: true
    ngram_size: 4
  lexical:
    enabled: true
    bm25_k1: 1.2
    bm25_b: 0.8
  hybrid:
    default_rrf_k: 50.0
embedder:
  enabled: true
  url: "http://localhost:8001/embed"
  timeout_seconds: 10
  worker_pool_size: 8
  channel_buffer_size: 512
  auto_embed_queries: true
`
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Server.GRPCPort != 50052 {
		t.Errorf("GRPCPort = %d, want 50052", cfg.Server.GRPCPort)
	}
	if cfg.Server.MetricsPort != 2113 {
		t.Errorf("MetricsPort = %d, want 2113", cfg.Server.MetricsPort)
	}
	if cfg.Store.MaxMemoryBytes != 67108864 {
		t.Errorf("MaxMemoryBytes = %d, want 67108864", cfg.Store.MaxMemoryBytes)
	}
	if cfg.Search.Surface.NgramSize != 4 {
		t.Errorf("NgramSize = %d, want 4", cfg.Search.Surface.NgramSize)
	}
	if cfg.Search.Lexical.BM25K1 != 1.2 {
		t.Errorf("BM25K1 = %f, want 1.2", cfg.Search.Lexical.BM25K1)
	}
	if cfg.Search.Hybrid.DefaultRRFK != 50.0 {
		t.Errorf("DefaultRRFK = %f, want 50.0", cfg.Search.Hybrid.DefaultRRFK)
	}
	if !cfg.Embedder.Enabled {
		t.Error("Embedder.Enabled should be true")
	}
	if cfg.Embedder.URL != "http://localhost:8001/embed" {
		t.Errorf("Embedder.URL = %s, want http://localhost:8001/embed", cfg.Embedder.URL)
	}
}

func TestLoad_NotFound_Fallback(t *testing.T) {
	cfg, err := config.Load("non_existent_file.yaml")
	if err != nil {
		t.Fatalf("expected fallback to default config on missing file, got error: %v", err)
	}
	if cfg.Server.GRPCPort != 50051 {
		t.Errorf("expected default GRPCPort 50051, got %d", cfg.Server.GRPCPort)
	}
	if cfg.Search.Surface.NgramSize != 3 {
		t.Errorf("expected default NgramSize 3, got %d", cfg.Search.Surface.NgramSize)
	}
}

func TestLoad_EnvOverrides(t *testing.T) {
	t.Setenv("GOGAGHE_SERVER_GRPC_PORT", "9999")
	t.Setenv("GOGAGHE_EMBEDDER_ENABLED", "true")
	t.Setenv("GOGAGHE_EMBEDDER_URL", "http://ai-host:9000/embed")
	t.Setenv("GOGAGHE_SEARCH_SURFACE_NGRAM_SIZE", "5")

	cfg, err := config.Load("")
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if cfg.Server.GRPCPort != 9999 {
		t.Errorf("GRPCPort = %d, want 9999", cfg.Server.GRPCPort)
	}
	if !cfg.Embedder.Enabled {
		t.Errorf("Embedder.Enabled = false, want true")
	}
	if cfg.Embedder.URL != "http://ai-host:9000/embed" {
		t.Errorf("Embedder.URL = %s, want http://ai-host:9000/embed", cfg.Embedder.URL)
	}
	if cfg.Search.Surface.NgramSize != 5 {
		t.Errorf("NgramSize = %d, want 5", cfg.Search.Surface.NgramSize)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	if cfg.Server.GRPCPort != 50051 {
		t.Errorf("default GRPCPort = %d, want 50051", cfg.Server.GRPCPort)
	}
	if cfg.Search.Hybrid.DefaultRRFK != 60.0 {
		t.Errorf("default DefaultRRFK = %f, want 60.0", cfg.Search.Hybrid.DefaultRRFK)
	}
	if cfg.Search.Lexical.BM25K1 != 1.5 {
		t.Errorf("default BM25K1 = %f, want 1.5", cfg.Search.Lexical.BM25K1)
	}
}
