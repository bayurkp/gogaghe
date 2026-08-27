// internal/config/config.go
package config

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config is the top-level application configuration.
type Config struct {
	Server   ServerConfig   `yaml:"server"`
	Store    StoreConfig    `yaml:"store"`
	Search   SearchConfig   `yaml:"search"`
	Embedder EmbedderConfig `yaml:"embedder"`
}

// ServerConfig holds network listener settings.
type ServerConfig struct {
	GRPCPort    int `yaml:"grpc_port"`
	MetricsPort int `yaml:"metrics_port"`
}

// StoreConfig holds in-memory engine tuning parameters.
type StoreConfig struct {
	MaxMemoryBytes          int64   `yaml:"max_memory_bytes"`
	TTLCheckIntervalSeconds int     `yaml:"ttl_check_interval_seconds"`
	EvictionThresholdRatio  float64 `yaml:"eviction_threshold_ratio"`
}

// SearchConfig holds tuning hyperparameters for search strategies and fusion.
type SearchConfig struct {
	Surface SurfaceConfig `yaml:"surface"`
	Lexical LexicalConfig `yaml:"lexical"`
	TFIDF   TFIDFConfig   `yaml:"tfidf"`
	LSA     LSAConfig     `yaml:"lsa"`
	Hybrid  HybridConfig  `yaml:"hybrid"`
}

// SurfaceConfig holds parameters for character n-gram surface search.
type SurfaceConfig struct {
	Enabled   bool `yaml:"enabled"`
	NgramSize int  `yaml:"ngram_size"` // default: 3
}

// LexicalConfig holds parameters for BM25 lexical search.
type LexicalConfig struct {
	Enabled bool    `yaml:"enabled"`
	BM25K1  float64 `yaml:"bm25_k1"` // default: 1.5
	BM25B   float64 `yaml:"bm25_b"`  // default: 0.75
}

// TFIDFConfig holds parameters for classic TF-IDF search.
type TFIDFConfig struct {
	Enabled bool `yaml:"enabled"`
}

// LSAConfig holds parameters for Latent Semantic Analysis (Truncated SVD).
type LSAConfig struct {
	Enabled bool `yaml:"enabled"`
	DimK    int  `yaml:"dim_k"` // default: 64
}

// HybridConfig holds parameters for reciprocal rank fusion.
type HybridConfig struct {
	DefaultRRFK float64 `yaml:"default_rrf_k"` // default: 60.0
}

// EmbedderConfig holds settings for the optional embedding sidecar.
type EmbedderConfig struct {
	Enabled           bool   `yaml:"enabled"`
	URL               string `yaml:"url"`
	TimeoutSeconds    int    `yaml:"timeout_seconds"`
	WorkerPoolSize    int    `yaml:"worker_pool_size"`
	ChannelBufferSize int    `yaml:"channel_buffer_size"`
	AutoEmbedQueries  bool   `yaml:"auto_embed_queries"` // auto-embed text queries if query_vector is omitted
}

// DefaultConfig returns a Config populated with production-ready fallback defaults.
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			GRPCPort:    50051,
			MetricsPort: 2112,
		},
		Store: StoreConfig{
			MaxMemoryBytes:          134217728, // 128 MiB
			TTLCheckIntervalSeconds: 30,
			EvictionThresholdRatio:  0.90,
		},
		Search: SearchConfig{
			Surface: SurfaceConfig{
				Enabled:   true,
				NgramSize: 3,
			},
			Lexical: LexicalConfig{
				Enabled: true,
				BM25K1:  1.5,
				BM25B:   0.75,
			},
			TFIDF: TFIDFConfig{
				Enabled: true,
			},
			LSA: LSAConfig{
				Enabled: true,
				DimK:    64,
			},
			Hybrid: HybridConfig{
				DefaultRRFK: 60.0,
			},
		},
		Embedder: EmbedderConfig{
			Enabled:           false,
			URL:               "http://localhost:8000/embed",
			TimeoutSeconds:    5,
			WorkerPoolSize:    4,
			ChannelBufferSize: 256,
			AutoEmbedQueries:  true,
		},
	}
}

// Load reads a YAML config file, applies defaults for missing fields, and overlays environment variables.
// If the file does not exist, it falls back to DefaultConfig() with a warning log.
func Load(path string) (*Config, error) {
	cfg := DefaultConfig()

	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				slog.Warn("config: file not found, using defaults", "path", path)
			} else {
				return nil, fmt.Errorf("config: read file %q: %w", path, err)
			}
		} else {
			if err := yaml.Unmarshal(data, cfg); err != nil {
				return nil, fmt.Errorf("config: unmarshal: %w", err)
			}
		}
	}

	applyValidationAndFallbacks(cfg)
	applyEnvOverrides(cfg)

	return cfg, nil
}

// applyValidationAndFallbacks ensures no required field is zero or out of bounds.
func applyValidationAndFallbacks(cfg *Config) {
	if cfg.Server.GRPCPort <= 0 {
		cfg.Server.GRPCPort = 50051
	}
	if cfg.Server.MetricsPort <= 0 {
		cfg.Server.MetricsPort = 2112
	}
	if cfg.Store.MaxMemoryBytes <= 0 {
		cfg.Store.MaxMemoryBytes = 134217728 // 128 MiB
	}
	if cfg.Store.TTLCheckIntervalSeconds <= 0 {
		cfg.Store.TTLCheckIntervalSeconds = 30
	}
	if cfg.Store.EvictionThresholdRatio <= 0 || cfg.Store.EvictionThresholdRatio > 1.0 {
		cfg.Store.EvictionThresholdRatio = 0.90
	}
	if cfg.Search.Surface.NgramSize <= 0 {
		cfg.Search.Surface.NgramSize = 3
	}
	if cfg.Search.Lexical.BM25K1 <= 0 {
		cfg.Search.Lexical.BM25K1 = 1.5
	}
	if cfg.Search.Lexical.BM25B <= 0 {
		cfg.Search.Lexical.BM25B = 0.75
	}
	if cfg.Search.LSA.DimK <= 0 {
		cfg.Search.LSA.DimK = 64
	}
	if cfg.Search.Hybrid.DefaultRRFK <= 0 {
		cfg.Search.Hybrid.DefaultRRFK = 60.0
	}
	if cfg.Embedder.URL == "" {
		cfg.Embedder.URL = "http://localhost:8000/embed"
	}
	if cfg.Embedder.TimeoutSeconds <= 0 {
		cfg.Embedder.TimeoutSeconds = 5
	}
	if cfg.Embedder.WorkerPoolSize <= 0 {
		cfg.Embedder.WorkerPoolSize = 4
	}
	if cfg.Embedder.ChannelBufferSize <= 0 {
		cfg.Embedder.ChannelBufferSize = 256
	}
}

// applyEnvOverrides checks standard environment variables and overrides matching config fields.
func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("GOGAGHE_SERVER_GRPC_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil && p > 0 {
			cfg.Server.GRPCPort = p
		}
	}
	if v := os.Getenv("GOGAGHE_SERVER_METRICS_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil && p > 0 {
			cfg.Server.MetricsPort = p
		}
	}
	if v := os.Getenv("GOGAGHE_STORE_MAX_MEMORY_BYTES"); v != "" {
		if m, err := strconv.ParseInt(v, 10, 64); err == nil && m > 0 {
			cfg.Store.MaxMemoryBytes = m
		}
	}
	if v := os.Getenv("GOGAGHE_EMBEDDER_ENABLED"); v != "" {
		cfg.Embedder.Enabled = strings.EqualFold(v, "true") || v == "1"
	}
	if v := os.Getenv("GOGAGHE_EMBEDDER_URL"); v != "" {
		cfg.Embedder.URL = v
	}
	if v := os.Getenv("GOGAGHE_SEARCH_SURFACE_NGRAM_SIZE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Search.Surface.NgramSize = n
		}
	}
	if v := os.Getenv("GOGAGHE_SEARCH_LSA_DIM_K"); v != "" {
		if k, err := strconv.Atoi(v); err == nil && k > 0 {
			cfg.Search.LSA.DimK = k
		}
	}
	if v := os.Getenv("GOGAGHE_SEARCH_HYBRID_DEFAULT_RRF_K"); v != "" {
		if k, err := strconv.ParseFloat(v, 64); err == nil && k > 0 {
			cfg.Search.Hybrid.DefaultRRFK = k
		}
	}
}
