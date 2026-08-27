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
	k1            float64
	b             float64
	invertedIndex map[string]map[string]int // token -> key -> term frequency
	docLengths    map[string]int            // key -> number of tokens
	docTokens     map[string][]string       // key -> token list for quick incremental removal
	avgDocLength  float64
	docCount      int
	totalTokens   int
}

// NewBM25Index creates an empty BM25Index with default parameters (k1=1.5, b=0.75).
func NewBM25Index() *BM25Index {
	return NewBM25IndexWithParams(bm25K1, bm25B)
}

// NewBM25IndexWithParams creates an empty BM25Index with custom k1 and b parameters.
func NewBM25IndexWithParams(k1, b float64) *BM25Index {
	if k1 <= 0 {
		k1 = bm25K1
	}
	if b <= 0 {
		b = bm25B
	}
	return &BM25Index{
		k1:            k1,
		b:             b,
		invertedIndex: make(map[string]map[string]int),
		docLengths:    make(map[string]int),
		docTokens:     make(map[string][]string),
	}
}

// Rebuild rebuilds the entire index from a snapshot of store items.
func (b *BM25Index) Rebuild(items map[string]Item) {
	b.invertedIndex = make(map[string]map[string]int)
	b.docLengths = make(map[string]int)
	b.docTokens = make(map[string][]string)
	b.totalTokens = 0
	b.docCount = 0

	for key, item := range items {
		b.IndexDocument(key, item.Value)
	}
}

// IndexDocument incrementally indexes or updates a single document.
func (b *BM25Index) IndexDocument(key string, value []byte) {
	// If key already existed, remove previous occurrences first
	if prevTokens, exists := b.docTokens[key]; exists {
		b.totalTokens -= len(prevTokens)
		for _, tok := range prevTokens {
			if postings, ok := b.invertedIndex[tok]; ok {
				delete(postings, key)
				if len(postings) == 0 {
					delete(b.invertedIndex, tok)
				}
			}
		}
	} else {
		b.docCount++
	}

	tokens := tokenize(string(value))
	b.docLengths[key] = len(tokens)
	b.docTokens[key] = tokens
	b.totalTokens += len(tokens)

	for _, tok := range tokens {
		if b.invertedIndex[tok] == nil {
			b.invertedIndex[tok] = make(map[string]int)
		}
		b.invertedIndex[tok][key]++
	}

	if b.docCount > 0 {
		b.avgDocLength = float64(b.totalTokens) / float64(b.docCount)
	} else {
		b.avgDocLength = 0
	}
}

// RemoveDocument incrementally removes a document from the index.
func (b *BM25Index) RemoveDocument(key string) {
	prevTokens, exists := b.docTokens[key]
	if !exists {
		return
	}

	b.totalTokens -= len(prevTokens)
	b.docCount--
	delete(b.docLengths, key)
	delete(b.docTokens, key)

	for _, tok := range prevTokens {
		if postings, ok := b.invertedIndex[tok]; ok {
			delete(postings, key)
			if len(postings) == 0 {
				delete(b.invertedIndex, tok)
			}
		}
	}

	if b.docCount > 0 {
		b.avgDocLength = float64(b.totalTokens) / float64(b.docCount)
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
				norm = float64(tf) * (b.k1 + 1) /
					(float64(tf) + b.k1*(1-b.b+b.b*dl/b.avgDocLength))
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
