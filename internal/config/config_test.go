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
  grpc_port: 50051
  metrics_port: 2112
store:
  max_memory_bytes: 134217728
  ttl_check_interval_seconds: 30
  eviction_threshold_ratio: 0.90
embedder:
  enabled: false
  url: "http://localhost:8000/embed"
  timeout_seconds: 5
  worker_pool_size: 4
  channel_buffer_size: 256
`
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Server.GRPCPort != 50051 {
		t.Errorf("GRPCPort = %d, want 50051", cfg.Server.GRPCPort)
	}
	if cfg.Server.MetricsPort != 2112 {
		t.Errorf("MetricsPort = %d, want 2112", cfg.Server.MetricsPort)
	}
	if cfg.Store.MaxMemoryBytes != 134217728 {
		t.Errorf("MaxMemoryBytes = %d, want 134217728", cfg.Store.MaxMemoryBytes)
	}
	if cfg.Store.TTLCheckIntervalSeconds != 30 {
		t.Errorf("TTLCheckIntervalSeconds = %d, want 30", cfg.Store.TTLCheckIntervalSeconds)
	}
	if cfg.Store.EvictionThresholdRatio != 0.90 {
		t.Errorf("EvictionThresholdRatio = %f, want 0.90", cfg.Store.EvictionThresholdRatio)
	}
	if cfg.Embedder.Enabled {
		t.Error("Embedder.Enabled should be false")
	}
	if cfg.Embedder.URL != "http://localhost:8000/embed" {
		t.Errorf("Embedder.URL = %s, want http://localhost:8000/embed", cfg.Embedder.URL)
	}
}

func TestLoad_NotFound(t *testing.T) {
	_, err := config.Load("non_existent_file.yaml")
	if err == nil {
		t.Error("expected error for non-existent file, got nil")
	}
}
