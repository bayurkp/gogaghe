// internal/store/bm25.go
package store

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

const (
	bm25K1 = 1.5
	bm25B  = 0.75
)

// ScoredKey pairs a store key with a ranking score.
type ScoredKey struct {
	Key   string
	Score float64
}

// BM25Index is an in-memory inverted index with BM25 scoring.
type BM25Index struct {
	invertedIndex map[string]map[string]int // token -> key -> term frequency
	docLengths    map[string]int            // key -> number of tokens
	avgDocLength  float64
	docCount      int
}

// NewBM25Index creates an empty BM25Index.
func NewBM25Index() *BM25Index {
	return &BM25Index{
		invertedIndex: make(map[string]map[string]int),
		docLengths:    make(map[string]int),
	}
}

// Rebuild rebuilds the entire index from a snapshot of store items.
func (b *BM25Index) Rebuild(items map[string]Item) {
	b.invertedIndex = make(map[string]map[string]int)
	b.docLengths = make(map[string]int)
	totalTokens := 0

	for key, item := range items {
		tokens := tokenize(string(item.Value))
		b.docLengths[key] = len(tokens)
		totalTokens += len(tokens)
		for _, tok := range tokens {
			if b.invertedIndex[tok] == nil {
				b.invertedIndex[tok] = make(map[string]int)
			}
			b.invertedIndex[tok][key]++
		}
	}
	b.docCount = len(items)
	if b.docCount > 0 {
		b.avgDocLength = float64(totalTokens) / float64(b.docCount)
	} else {
		b.avgDocLength = 0
	}
}

// Search returns the top-k documents scored by BM25 for a query string.
func (b *BM25Index) Search(query string, topK int) []ScoredKey {
	qTokens := tokenize(query)
	scores := make(map[string]float64)

	for _, tok := range qTokens {
		postings, ok := b.invertedIndex[tok]
		if !ok {
			continue
		}
		df := float64(len(postings))
		idf := math.Log((float64(b.docCount)-df+0.5)/(df+0.5) + 1)
		for key, tf := range postings {
			dl := float64(b.docLengths[key])
			var norm float64
			if b.avgDocLength > 0 {
				norm = float64(tf) * (bm25K1 + 1) /
					(float64(tf) + bm25K1*(1-bm25B+bm25B*dl/b.avgDocLength))
			} else {
				norm = float64(tf)
			}
			scores[key] += idf * norm
		}
	}

	ranked := make([]ScoredKey, 0, len(scores))
	for k, s := range scores {
		ranked = append(ranked, ScoredKey{Key: k, Score: s})
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

// tokenize lowercases and splits text into alphanumeric tokens.
func tokenize(text string) []string {
	lower := strings.ToLower(text)
	return strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
