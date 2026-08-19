// internal/server/metrics.go
package server

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all Prometheus collectors for gogaghe.
type Metrics struct {
	CacheHits         prometheus.Counter
	CacheMisses       prometheus.Counter
	OperationDuration *prometheus.HistogramVec
	MemoryUsageBytes  prometheus.Gauge
	ItemsCount        prometheus.Gauge
	GoroutinesActive  prometheus.Gauge

	registry *prometheus.Registry
}

// NewMetrics creates and registers all metrics with an isolated registry.
func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()

	m := &Metrics{
		registry: reg,
		CacheHits: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "gogaghe_cache_hits_total",
			Help: "Total number of cache hits.",
		}),
		CacheMisses: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "gogaghe_cache_misses_total",
			Help: "Total number of cache misses.",
		}),
		OperationDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gogaghe_operation_duration_seconds",
			Help:    "Duration of store operations.",
			Buckets: prometheus.DefBuckets,
		}, []string{"operation"}),
		MemoryUsageBytes: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gogaghe_memory_usage_bytes",
			Help: "Approximate memory used by stored items.",
		}),
		ItemsCount: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gogaghe_items_count",
			Help: "Number of items currently in the store.",
		}),
		GoroutinesActive: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gogaghe_goroutines_active",
			Help: "Number of active goroutines.",
		}),
	}

	reg.MustRegister(
		m.CacheHits,
		m.CacheMisses,
		m.OperationDuration,
		m.MemoryUsageBytes,
		m.ItemsCount,
		m.GoroutinesActive,
	)
	return m
}

// StartHTTPServer starts the /metrics HTTP endpoint on the given port.
func (m *Metrics) StartHTTPServer(port int) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{}))
	addr := fmt.Sprintf(":%d", port)
	return http.ListenAndServe(addr, mux)
}
