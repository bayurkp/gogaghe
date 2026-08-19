// internal/store/eviction.go
package store

import "container/heap"

// evictionCandidate is an item eligible for eviction.
type evictionCandidate struct {
	key      string
	priority float64 // lower = evict first
	index    int     // heap index
}

// evictionHeap is a min-heap of evictionCandidates ordered by priority.
type evictionHeap []*evictionCandidate

func (h evictionHeap) Len() int           { return len(h) }
func (h evictionHeap) Less(i, j int) bool { return h[i].priority < h[j].priority }
func (h evictionHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}
func (h *evictionHeap) Push(x any) {
	c := x.(*evictionCandidate)
	c.index = len(*h)
	*h = append(*h, c)
}
func (h *evictionHeap) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return c
}

// Evict removes items from the engine until MemoryUsageBytes <= targetBytes.
// Eviction priority is computed as CostMs / (AccessCount + 1):
// items with lower priority are evicted first.
func Evict(e *Engine, targetBytes int64) {
	for e.MemoryUsageBytes() > targetBytes {
		items := e.Items()
		if len(items) == 0 {
			break
		}

		h := make(evictionHeap, 0, len(items))
		for key, item := range items {
			priority := float64(item.CostMs) / float64(item.AccessCount+1)
			h = append(h, &evictionCandidate{key: key, priority: priority})
		}
		heap.Init(&h)

		if h.Len() == 0 {
			break
		}
		candidate := heap.Pop(&h).(*evictionCandidate)
		e.Delete(candidate.key)
	}
}
