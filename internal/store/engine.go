// internal/store/engine.go
package store

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// Item is the value stored for each cache key.
type Item struct {
	Value          []byte
	CostMs         int64
	Vector         []float32
	AccessCount    int64
	LastAccessedAt time.Time
	ExpiresAt      time.Time // zero = no expiry
}

// itemSize estimates the heap footprint of a single item in bytes.
func itemSize(key string, item Item) int64 {
	s := int64(len(key)) + int64(len(item.Value)) + int64(len(item.Vector)*4)
	s += int64(unsafe.Sizeof(item))
	return s
}

// Engine is a thread-safe in-memory key-value store.
type Engine struct {
	mu             sync.RWMutex
	data           map[string]Item
	maxMemoryBytes int64
	memoryUsed     int64 // tracked atomically
	stopCh         chan struct{}
	ttlTicker      *time.Ticker
}

// NewEngine creates a new Engine and starts the TTL background worker.
func NewEngine(maxMemoryBytes int64, ttlCheckInterval time.Duration) *Engine {
	e := &Engine{
		data:           make(map[string]Item),
		maxMemoryBytes: maxMemoryBytes,
		stopCh:         make(chan struct{}),
		ttlTicker:      time.NewTicker(ttlCheckInterval),
	}
	go e.ttlWorker()
	return e
}

// Set stores a key-value item. Returns an error if memory limit is exceeded.
func (e *Engine) Set(key string, item Item) error {
	size := itemSize(key, item)
	e.mu.Lock()
	defer e.mu.Unlock()

	var oldSize int64
	if old, exists := e.data[key]; exists {
		oldSize = itemSize(key, old)
	}

	netDelta := size - oldSize
	if atomic.LoadInt64(&e.memoryUsed)+netDelta > e.maxMemoryBytes {
		return fmt.Errorf("store: memory limit exceeded (limit=%d bytes)", e.maxMemoryBytes)
	}

	e.data[key] = item
	atomic.AddInt64(&e.memoryUsed, netDelta)
	return nil
}

// Get retrieves an item and increments its access counter.
// Returns false if the key does not exist or has expired.
func (e *Engine) Get(key string) (Item, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()

	item, ok := e.data[key]
	if !ok {
		return Item{}, false
	}
	if !item.ExpiresAt.IsZero() && time.Now().After(item.ExpiresAt) {
		delete(e.data, key)
		atomic.AddInt64(&e.memoryUsed, -itemSize(key, item))
		return Item{}, false
	}
	item.AccessCount++
	item.LastAccessedAt = time.Now()
	e.data[key] = item
	return item, true
}

// Delete removes a key. Returns true if the key existed.
func (e *Engine) Delete(key string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()

	item, ok := e.data[key]
	if ok {
		delete(e.data, key)
		atomic.AddInt64(&e.memoryUsed, -itemSize(key, item))
	}
	return ok
}

// Len returns the number of items currently stored.
func (e *Engine) Len() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.data)
}

// MemoryUsageBytes returns the approximate memory used by stored items.
func (e *Engine) MemoryUsageBytes() int64 {
	return atomic.LoadInt64(&e.memoryUsed)
}

// Items returns a snapshot copy of all stored items (safe for concurrent use).
func (e *Engine) Items() map[string]Item {
	e.mu.RLock()
	defer e.mu.RUnlock()

	snapshot := make(map[string]Item, len(e.data))
	for k, v := range e.data {
		snapshot[k] = v
	}
	return snapshot
}

// Stop terminates the TTL background worker.
func (e *Engine) Stop() {
	e.ttlTicker.Stop()
	close(e.stopCh)
}

// ttlWorker scans for expired keys on each tick.
func (e *Engine) ttlWorker() {
	for {
		select {
		case <-e.stopCh:
			return
		case <-e.ttlTicker.C:
			e.evictExpired()
		}
	}
}

func (e *Engine) evictExpired() {
	now := time.Now()
	e.mu.Lock()
	defer e.mu.Unlock()

	for key, item := range e.data {
		if !item.ExpiresAt.IsZero() && now.After(item.ExpiresAt) {
			delete(e.data, key)
			atomic.AddInt64(&e.memoryUsed, -itemSize(key, item))
		}
	}
}
