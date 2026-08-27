# Design Specification: TF-IDF & LSA/LSI Hierarchical Search Integration

## Overview
Expand `gogaghe` search capabilities by integrating classic statistical Information Retrieval (IR) models without requiring external neural network sidecars or Cgo dependencies.

The search strategies follow a **5-tier hierarchical paradigm** ordered by mathematical abstraction depth:
1. `SEARCH_STRATEGY_SURFACE_TRIGRAM`: Character-level typo/substring tolerance.
2. `SEARCH_STRATEGY_LEXICAL_BM25`: Word-level probabilistic relevance.
3. `SEARCH_STRATEGY_LEXICAL_TFIDF`: Word-level vector space model (Salton).
4. `SEARCH_STRATEGY_SEMANTIC_LSA`: Statistical concept-level latent semantics via Truncated SVD.
5. `SEARCH_STRATEGY_SEMANTIC_DENSE`: Deep neural embeddings via dense float vectors.

---

## 1. Protobuf API Contract (`api/proto/gogaghe/v1/gogaghe.proto`)

### Enum Definition
```protobuf
enum SearchStrategy {
  SEARCH_STRATEGY_UNSPECIFIED     = 0;
  SEARCH_STRATEGY_SURFACE_TRIGRAM = 1;
  SEARCH_STRATEGY_LEXICAL_BM25    = 2;
  SEARCH_STRATEGY_LEXICAL_TFIDF   = 3;
  SEARCH_STRATEGY_SEMANTIC_LSA    = 4;
  SEARCH_STRATEGY_SEMANTIC_DENSE  = 5;
}
```

### Dedicated RPCs
- `SurfaceSearch(SurfaceSearchRequest) returns (SurfaceSearchResponse)`
- `Bm25Search(Bm25SearchRequest) returns (Bm25SearchResponse)` (aliased/standardized from LexicalSearch)
- `TfidfSearch(TfidfSearchRequest) returns (TfidfSearchResponse)`
- `LsaSearch(LsaSearchRequest) returns (LsaSearchResponse)`
- `VectorSearch(VectorSearchRequest) returns (VectorSearchResponse)`
- `HybridSearch(HybridSearchRequest) returns (HybridSearchResponse)`

---

## 2. Storage & Algorithms (`internal/store/`)

### A. TF-IDF Engine (`internal/store/tfidf.go`)
- **Indexing**:
  - Build vocabulary map `token -> termID`.
  - Maintain document frequencies `df` and term frequencies `tf`.
  - Compute sublinear TF: $\text{tf\_weight} = 1 + \ln(\text{tf})$.
  - Smooth IDF: $\text{idf} = \ln\left(\frac{N - \text{df} + 0.5}{\text{df} + 0.5} + 1\right)$.
  - Pre-calculate $L_2$ document norms for instant unit normalization.
- **Query Processing**:
  - Convert query text into sparse term vector.
  - Calculate Cosine Similarity with all matching documents.

### B. LSA Engine (`internal/store/lsa.go`)
- **Pure-Go Truncated SVD**:
  - Factorizes the term-document TF-IDF matrix $A \approx U_k \Sigma_k V_k^T$.
  - Uses pure-Go Power Iteration / Lanczos orthogonalization (Zero Cgo, zero external dependencies).
  - Truncated dimension $k$ configurable (default $k=64$).
- **Document Projection**:
  - Low-rank document vectors: $D_k = \Sigma_k V_k^T$ ($k \times N$ matrix).
- **Query Projection & Search**:
  - Query vector projected into latent semantic space: $\hat{q} = q^T U_k \Sigma_k^{-1}$.
  - Computes Cosine Similarity between $\hat{q}$ and low-rank document vectors $D_k$.

### C. Reciprocal Rank Fusion (`internal/store/hybrid.go`)
- Seamlessly fuses any combination of the 5 strategies:
  $$\text{RRF Score}(d) = \sum_{s \in \text{Strategies}} \frac{1}{k_{\text{rrf}} + \text{rank}_s(d)}$$

---

## 3. Engineering Constraints & Compliance
- **Zero Cgo**: Compiles with `CGO_ENABLED=0`.
- **Thread Safety**: Snapshot isolation via `Engine.Items()`.
- **Concurrency**: Goroutine worker pools for matrix dot-products and Cosine Similarity.
- **Race Free**: Passes `go test -v -race ./...`.
