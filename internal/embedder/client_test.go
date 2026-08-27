// internal/embedder/client_test.go
package embedder_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/bayurkp/gogaghe/internal/config"
	"github.com/bayurkp/gogaghe/internal/embedder"
)

func TestClient_EnqueueAndCallback(t *testing.T) {
	expectedVector := []float32{0.1, 0.2, 0.3}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var payload struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		resp := struct {
			Vector []float32 `json:"vector"`
		}{
			Vector: expectedVector,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	cfg := config.EmbedderConfig{
		Enabled:           true,
		URL:               ts.URL,
		TimeoutSeconds:    2,
		WorkerPoolSize:    2,
		ChannelBufferSize: 10,
	}

	var mu sync.Mutex
	received := make(map[string][]float32)
	doneCh := make(chan struct{}, 1)

	client := embedder.NewClientWithCallback(cfg, func(key string, vec []float32) {
		mu.Lock()
		received[key] = vec
		if len(received) == 1 {
			select {
			case doneCh <- struct{}{}:
			default:
			}
		}
		mu.Unlock()
	})

	client.Enqueue(embedder.EmbedRequest{Key: "k1", Text: "sample text"})

	select {
	case <-doneCh:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for embed callback")
	}

	client.Stop()

	mu.Lock()
	vec, ok := received["k1"]
	mu.Unlock()

	if !ok {
		t.Fatal("key k1 not received in callback")
	}
	if len(vec) != len(expectedVector) {
		t.Fatalf("vector len = %d, want %d", len(vec), len(expectedVector))
	}
}

func TestClient_Embed(t *testing.T) {
	expectedVector := []float32{0.5, 0.6, 0.7}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := struct {
			Vector []float32 `json:"vector"`
		}{
			Vector: expectedVector,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	cfg := config.EmbedderConfig{
		Enabled:        true,
		URL:            ts.URL,
		TimeoutSeconds: 2,
	}

	client := embedder.NewClient(cfg)
	defer client.Stop()

	vec, err := client.Embed(rContext(), "sepatu putih")
	if err != nil {
		t.Fatalf("Embed() error: %v", err)
	}
	if len(vec) != len(expectedVector) {
		t.Fatalf("Embed() vector len = %d, want %d", len(vec), len(expectedVector))
	}
	if vec[0] != 0.5 || vec[1] != 0.6 || vec[2] != 0.7 {
		t.Errorf("Embed() vector = %v, want %v", vec, expectedVector)
	}
}

func TestClient_ChannelFullDrop(t *testing.T) {
	// Dummy URL (server not called because channel is full and we drop)
	cfg := config.EmbedderConfig{
		Enabled:           true,
		URL:               "http://invalid.local",
		TimeoutSeconds:    1,
		WorkerPoolSize:    0, // 0 workers so channel never drains
		ChannelBufferSize: 1,
	}

	client := embedder.NewClient(cfg)
	// First enqueue fits in buffer
	client.Enqueue(embedder.EmbedRequest{Key: "k1", Text: "text1"})
	// Second enqueue should be dropped non-blocking without hang
	client.Enqueue(embedder.EmbedRequest{Key: "k2", Text: "text2"})
	client.Stop()
}

func rContext() context.Context {
	return context.Background()
}
