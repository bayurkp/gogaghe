// internal/config/config.go
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the top-level application configuration.
type Config struct {
	Server   ServerConfig   `yaml:"server"`
	Store    StoreConfig    `yaml:"store"`
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

// EmbedderConfig holds settings for the optional embedding sidecar.
type EmbedderConfig struct {
	Enabled           bool   `yaml:"enabled"`
	URL               string `yaml:"url"`
	TimeoutSeconds    int    `yaml:"timeout_seconds"`
	WorkerPoolSize    int    `yaml:"worker_pool_size"`
	ChannelBufferSize int    `yaml:"channel_buffer_size"`
}

// Load reads a YAML config file and returns a parsed Config.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config: read file %q: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("config: unmarshal: %w", err)
	}
	return &cfg, nil
}
