// internal/server/metrics_test.go
package server_test

import (
	"testing"

	"github.com/bayurkp/gogaghe/internal/server"
)

func TestNewMetrics(t *testing.T) {
	m := server.NewMetrics()
	if m == nil {
		t.Fatal("expected non-nil Metrics")
	}

	// Verify all collectors can be manipulated without panic
	m.CacheHits.Inc()
	m.CacheMisses.Inc()
	m.OperationDuration.WithLabelValues("get").Observe(0.001)
	m.MemoryUsageBytes.Set(1024)
	m.ItemsCount.Set(5)
	m.GoroutinesActive.Set(10)
}
