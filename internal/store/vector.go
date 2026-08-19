// internal/store/vector.go
package store

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// CosineSimilarity computes cosine similarity between two equal-length float32 vectors.
// Returns values in [-1.0, 1.0]. Returns 0.0 for zero vectors or dimension mismatch.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// vectorJob is a unit of work dispatched to a worker goroutine.
type vectorJob struct {
	key   string
	item  Item
	query []float32
}

// vectorResult carries the scored output from a worker.
type vectorResult struct {
	key   string
	score float64
}

// VectorSearch performs parallel cosine similarity search over items using a
// goroutine worker pool sized to GOMAXPROCS. Only items with a non-nil Vector
// whose length matches queryVec are considered.
func VectorSearch(queryVec []float32, items map[string]Item, topK int) []ScoredKey {
	if len(items) == 0 || len(queryVec) == 0 {
		return nil
	}

	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers < 1 {
		numWorkers = 1
	}

	jobs := make(chan vectorJob, len(items))
	results := make(chan vectorResult, len(items))

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				if len(job.item.Vector) != len(job.query) {
					continue
				}
				score := CosineSimilarity(job.item.Vector, job.query)
				results <- vectorResult{key: job.key, score: score}
			}
		}()
	}

	for key, item := range items {
		if len(item.Vector) > 0 {
			jobs <- vectorJob{key: key, item: item, query: queryVec}
		}
	}
	close(jobs)

	wg.Wait()
	close(results)

	ranked := make([]ScoredKey, 0, len(items))
	for r := range results {
		ranked = append(ranked, ScoredKey{Key: r.key, Score: r.score})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].Score == ranked[j].Score {
			return ranked[i].Key < ranked[j].Key
		}
		return ranked[i].Score > ranked[j].Score
	})
	if topK > 0 && len(ranked) > topK {
		ranked = ranked[:topK]
	}
	return ranked
}
