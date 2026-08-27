// internal/store/tfidf.go
package store

import (
	"math"
	"sort"
)

// TFIDFIndex is an in-memory inverted index with Vector Space Model (Salton TF-IDF) scoring.
type TFIDFIndex struct {
	invertedIndex map[string]map[string]int // token -> key -> term frequency
	docTokens     map[string][]string       // key -> list of tokens
	docNorms      map[string]float64       // key -> L2 norm of document TF-IDF vector
	docCount      int
}

// NewTFIDFIndex creates an empty TFIDFIndex.
func NewTFIDFIndex() *TFIDFIndex {
	return &TFIDFIndex{
		invertedIndex: make(map[string]map[string]int),
		docTokens:     make(map[string][]string),
		docNorms:      make(map[string]float64),
	}
}

// Rebuild rebuilds the entire TF-IDF index from a snapshot of store items.
func (t *TFIDFIndex) Rebuild(items map[string]Item) {
	t.invertedIndex = make(map[string]map[string]int)
	t.docTokens = make(map[string][]string)
	t.docNorms = make(map[string]float64)
	t.docCount = 0

	for key, item := range items {
		tokens := tokenize(string(item.Value))
		if len(tokens) == 0 {
			continue
		}
		t.docCount++
		t.docTokens[key] = tokens
		for _, tok := range tokens {
			if t.invertedIndex[tok] == nil {
				t.invertedIndex[tok] = make(map[string]int)
			}
			t.invertedIndex[tok][key]++
		}
	}

	// Calculate document L2 norms after full vocabulary & docCount are established
	t.recomputeNorms()
}

// IndexDocument incrementally indexes or updates a single document.
func (t *TFIDFIndex) IndexDocument(key string, value []byte) {
	// If key already existed, remove previous occurrences first
	if prevTokens, exists := t.docTokens[key]; exists {
		for _, tok := range prevTokens {
			if postings, ok := t.invertedIndex[tok]; ok {
				delete(postings, key)
				if len(postings) == 0 {
					delete(t.invertedIndex, tok)
				}
			}
		}
	} else {
		t.docCount++
	}

	tokens := tokenize(string(value))
	t.docTokens[key] = tokens

	for _, tok := range tokens {
		if t.invertedIndex[tok] == nil {
			t.invertedIndex[tok] = make(map[string]int)
		}
		t.invertedIndex[tok][key]++
	}

	t.recomputeNorms()
}

// RemoveDocument incrementally removes a document from the index.
func (t *TFIDFIndex) RemoveDocument(key string) {
	prevTokens, exists := t.docTokens[key]
	if !exists {
		return
	}

	t.docCount--
	delete(t.docTokens, key)
	delete(t.docNorms, key)

	for _, tok := range prevTokens {
		if postings, ok := t.invertedIndex[tok]; ok {
			delete(postings, key)
			if len(postings) == 0 {
				delete(t.invertedIndex, tok)
			}
		}
	}

	t.recomputeNorms()
}

// recomputeNorms recalculates L2 norm for all indexed documents.
func (t *TFIDFIndex) recomputeNorms() {
	t.docNorms = make(map[string]float64, len(t.docTokens))
	if t.docCount == 0 {
		return
	}

	for key, tokens := range t.docTokens {
		tfMap := make(map[string]int)
		for _, tok := range tokens {
			tfMap[tok]++
		}

		var sumSq float64
		for tok, count := range tfMap {
			df := float64(len(t.invertedIndex[tok]))
			idf := math.Log((float64(t.docCount)-df+0.5)/(df+0.5) + 1)
			tfWeight := 1.0 + math.Log(float64(count))
			w := tfWeight * idf
			sumSq += w * w
		}
		t.docNorms[key] = math.Sqrt(sumSq)
	}
}

// Search returns top-k documents scored by TF-IDF Cosine Similarity against the query string.
func (t *TFIDFIndex) Search(query string, topK int) []ScoredKey {
	qTokens := tokenize(query)
	if len(qTokens) == 0 || t.docCount == 0 {
		return nil
	}

	qTfMap := make(map[string]int)
	for _, tok := range qTokens {
		qTfMap[tok]++
	}

	dotProducts := make(map[string]float64)
	var qNormSq float64

	for tok, qCount := range qTfMap {
		postings, ok := t.invertedIndex[tok]
		if !ok {
			continue
		}
		df := float64(len(postings))
		idf := math.Log((float64(t.docCount)-df+0.5)/(df+0.5) + 1)
		qWeight := (1.0 + math.Log(float64(qCount))) * idf
		qNormSq += qWeight * qWeight

		for docKey, docCount := range postings {
			dWeight := (1.0 + math.Log(float64(docCount))) * idf
			dotProducts[docKey] += qWeight * dWeight
		}
	}

	qNorm := math.Sqrt(qNormSq)
	if qNorm == 0 {
		return nil
	}

	ranked := make([]ScoredKey, 0, len(dotProducts))
	for docKey, dot := range dotProducts {
		dNorm := t.docNorms[docKey]
		if dNorm == 0 {
			continue
		}
		cosine := dot / (qNorm * dNorm)
		ranked = append(ranked, ScoredKey{Key: docKey, Score: cosine})
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
