// internal/embedder/client.go
package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/bayurkp/gogaghe/internal/config"
)

// EmbedRequest represents an item to be embedded by the sidecar.
type EmbedRequest struct {
	Key  string
	Text string
}

// EmbedCallback is called with the resulting vector after embedding completes.
type EmbedCallback func(key string, vector []float32)

// embedPayload is the JSON body sent to the embedding sidecar.
type embedPayload struct {
	Text string `json:"text"`
}

// embedResponse is the JSON body expected from the embedding sidecar.
type embedResponse struct {
	Vector []float32 `json:"vector"`
}

// Client dispatches embedding requests to the sidecar asynchronously.
type Client struct {
	cfg      config.EmbedderConfig
	jobCh    chan EmbedRequest
	callback EmbedCallback
	wg       sync.WaitGroup
	stopOnce sync.Once
	stopCh   chan struct{}
	http     *http.Client
}

// NewClient creates a Client with a no-op callback. Use NewClientWithCallback
// to receive embedded vectors.
func NewClient(cfg config.EmbedderConfig) *Client {
	return NewClientWithCallback(cfg, nil)
}

// NewClientWithCallback creates a Client that calls cb with each completed embedding.
// The callback is invoked from a worker goroutine — it must be goroutine-safe.
func NewClientWithCallback(cfg config.EmbedderConfig, cb EmbedCallback) *Client {
	bufSize := cfg.ChannelBufferSize
	if bufSize <= 0 {
		bufSize = 256
	}
	timeoutSec := cfg.TimeoutSeconds
	if timeoutSec <= 0 {
		timeoutSec = 5
	}
	c := &Client{
		cfg:      cfg,
		jobCh:    make(chan EmbedRequest, bufSize),
		callback: cb,
		stopCh:   make(chan struct{}),
		http: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
	}
	for i := 0; i < cfg.WorkerPoolSize; i++ {
		c.wg.Add(1)
		go c.worker()
	}
	return c
}

// Enqueue submits a text for async embedding. Drops the request silently if
// the buffer is full to avoid blocking the caller.
func (c *Client) Enqueue(req EmbedRequest) {
	select {
	case c.jobCh <- req:
	default:
		slog.Warn("embedder: channel full, dropping embed request", "key", req.Key)
	}
}

// Stop signals all workers to shut down and waits for them to finish.
func (c *Client) Stop() {
	c.stopOnce.Do(func() {
		close(c.stopCh)
		close(c.jobCh)
	})
	c.wg.Wait()
}

func (c *Client) worker() {
	defer c.wg.Done()
	for req := range c.jobCh {
		vec, err := c.Embed(context.Background(), req.Text)
		if err != nil {
			slog.Error("embedder: embed failed", "key", req.Key, "err", err)
			continue
		}
		if c.callback != nil {
			c.callback(req.Key, vec)
		}
	}
}

// Embed synchronously sends a text to the sidecar and returns its vector embedding.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(embedPayload{Text: text})
	if err != nil {
		return nil, err
	}

	timeout := time.Duration(c.cfg.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Second
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Vector, nil
}
