// internal/store/ngram.go
package store

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// NgramIndex is an in-memory inverted index of character n-grams
// designed for typo-tolerant, prefix, code, and surface similarity search.
type NgramIndex struct {
	n             int
	invertedIndex map[string]map[string]int // ngram -> key -> frequency
	docNgramCount map[string]int            // key -> total ngrams in document
	docNgrams     map[string][]string       // key -> slice of ngrams for fast removal
}

// NewNgramIndex creates a new NgramIndex with the specified n-gram size (defaults to 3).
func NewNgramIndex(n int) *NgramIndex {
	if n <= 0 {
		n = 3
	}
	return &NgramIndex{
		n:             n,
		invertedIndex: make(map[string]map[string]int),
		docNgramCount: make(map[string]int),
		docNgrams:     make(map[string][]string),
	}
}

// Rebuild reconstructs the character n-gram inverted index from a snapshot of store items.
func (idx *NgramIndex) Rebuild(items map[string]Item) {
	idx.invertedIndex = make(map[string]map[string]int)
	idx.docNgramCount = make(map[string]int)
	idx.docNgrams = make(map[string][]string)

	for key, item := range items {
		idx.IndexDocument(key, item.Value)
	}
}

// IndexDocument incrementally indexes a document key and its value.
func (idx *NgramIndex) IndexDocument(key string, value []byte) {
	// Remove previous n-grams if key existed
	if prevNgrams, exists := idx.docNgrams[key]; exists {
		for _, ng := range prevNgrams {
			if postings, ok := idx.invertedIndex[ng]; ok {
				delete(postings, key)
				if len(postings) == 0 {
					delete(idx.invertedIndex, ng)
				}
			}
		}
		delete(idx.docNgramCount, key)
		delete(idx.docNgrams, key)
	}

	ngrams := ExtractNgrams(string(value), idx.n)
	if len(ngrams) == 0 {
		return
	}

	idx.docNgramCount[key] = len(ngrams)
	idx.docNgrams[key] = ngrams

	for _, ng := range ngrams {
		if idx.invertedIndex[ng] == nil {
			idx.invertedIndex[ng] = make(map[string]int)
		}
		idx.invertedIndex[ng][key]++
	}
}

// RemoveDocument incrementally removes a document from the n-gram index.
func (idx *NgramIndex) RemoveDocument(key string) {
	prevNgrams, exists := idx.docNgrams[key]
	if !exists {
		return
	}

	for _, ng := range prevNgrams {
		if postings, ok := idx.invertedIndex[ng]; ok {
			delete(postings, key)
			if len(postings) == 0 {
				delete(idx.invertedIndex, ng)
			}
		}
	}
	delete(idx.docNgramCount, key)
	delete(idx.docNgrams, key)
}

// Search computes character n-gram similarity (Sørensen–Dice coefficient)
// against all matching documents for a query and returns the top-k results.
func (idx *NgramIndex) Search(query string, topK int) []ScoredKey {
	qNgrams := ExtractNgrams(query, idx.n)
	if len(qNgrams) == 0 {
		return nil
	}

	// Count shared n-grams per document
	sharedCounts := make(map[string]int)
	qCounts := make(map[string]int)
	for _, ng := range qNgrams {
		qCounts[ng]++
	}

	for ng, qFreq := range qCounts {
		postings, ok := idx.invertedIndex[ng]
		if !ok {
			continue
		}
		for docKey, docFreq := range postings {
			// Min overlap for frequency-aware matching
			if qFreq < docFreq {
				sharedCounts[docKey] += qFreq
			} else {
				sharedCounts[docKey] += docFreq
			}
		}
	}

	if len(sharedCounts) == 0 {
		return nil
	}

	ranked := make([]ScoredKey, 0, len(sharedCounts))
	lenQ := float64(len(qNgrams))

	for docKey, shared := range sharedCounts {
		docLen := float64(idx.docNgramCount[docKey])
		// Dice coefficient: 2 * |Q ∩ D| / (|Q| + |D|)
		diceScore := (2.0 * float64(shared)) / (lenQ + docLen)
		if diceScore > 0 {
			ranked = append(ranked, ScoredKey{
				Key:   docKey,
				Score: math.Round(diceScore*10000) / 10000, // round to 4 decimals
			})
		}
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

// ExtractNgrams splits normalized text into lowercase character n-grams.
func ExtractNgrams(text string, n int) []string {
	if n <= 0 {
		n = 3
	}

	// Normalize: lowercase and replace non-alphanumerics with space
	var sb strings.Builder
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			sb.WriteRune(r)
		} else {
			sb.WriteRune(' ')
		}
	}

	words := strings.Fields(sb.String())
	var ngrams []string

	for _, word := range words {
		runes := []rune(word)
		if len(runes) == 0 {
			continue
		}
		if len(runes) < n {
			// For short tokens, keep the word itself as a single token unit
			ngrams = append(ngrams, string(runes))
			continue
		}
		for i := 0; i <= len(runes)-n; i++ {
			ngrams = append(ngrams, string(runes[i:i+n]))
		}
	}

	return ngrams
}
