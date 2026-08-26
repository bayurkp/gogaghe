// internal/store/hybrid.go
package store

import "sort"

// RRF merges two ranked lists using Reciprocal Rank Fusion.
// k is the RRF constant (typically 60.0). Higher k reduces the influence of
// high-rank outliers. Returns the top-k fused results.
func RRF(bm25Results, vectorResults []ScoredKey, topK int, k float64) []ScoredKey {
	return RRFMulti([][]ScoredKey{bm25Results, vectorResults}, topK, k)
}

// RRFMulti merges multiple ranked lists (e.g., Character n-gram, BM25 lexical, Dense Vector semantic)
// using Reciprocal Rank Fusion.
// RRF score(d) = Σ 1 / (k + rank_i(d))
func RRFMulti(rankLists [][]ScoredKey, topK int, k float64) []ScoredKey {
	if k <= 0 {
		k = 60.0
	}
	scores := make(map[string]float64)

	for _, list := range rankLists {
		for rank, item := range list {
			scores[item.Key] += 1.0 / (k + float64(rank+1))
		}
	}

	merged := make([]ScoredKey, 0, len(scores))
	for key, score := range scores {
		merged = append(merged, ScoredKey{Key: key, Score: score})
	}

	sort.Slice(merged, func(i, j int) bool {
		if merged[i].Score == merged[j].Score {
			return merged[i].Key < merged[j].Key
		}
		return merged[i].Score > merged[j].Score
	})

	if topK > 0 && len(merged) > topK {
		merged = merged[:topK]
	}
	return merged
}
