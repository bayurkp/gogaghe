# gogaghe — In-Memory Hybrid Store & Vector Engine

An in-memory key-value store with multi-resolution hybrid search (Character N-gram + BM25 + Dense Vector), cost-aware eviction, and async embedding pipeline. Pure Go, zero Cgo, single binary.

---

## 📌 Overview

**gogaghe** is a standalone in-memory microservice for low-latency key-value storage with built-in search. It combines three search strategies — character n-gram (surface similarity), BM25 (lexical relevance), and cosine similarity (semantic matching) — fused via Reciprocal Rank Fusion into a single ranked result.

Architecturally, gogaghe is a single Go binary that embeds all search indexes in-process. This differs from systems like Redis (which requires a separate server process) or managed vector databases like Pinecone (which require network calls to a cloud service). The trade-off is that gogaghe's data lives in a single process and is not distributed.

This project covers Go concurrency patterns, gRPC service design, in-memory data structures, hybrid search algorithms, and containerized microservice deployment.

---

## 🎯 Design Goals & Trade-offs

| Goal | Approach | Trade-off |
| :--- | :--- | :--- |
| **Embeddable search** — no separate server process | All indexes (N-gram, BM25, vector) live in-process | Data is not distributed; limited to single-node memory |
| **Typo-tolerant surface matching** | Character trigram index with Sørensen–Dice scoring | Additional memory for n-gram inverted index |
| **Multi-signal ranking** | 3-stream RRF (surface + lexical + semantic) | More CPU per query than single-strategy search |
| **Cost-aware eviction** | Priority = `CostMs / (AccessCount + 1)` | Requires callers to provide `cost_ms` on write |
| **Non-blocking writes with embedding** | Async sidecar pipeline; `Set` returns instantly | Vectors are eventually consistent (not immediate) |
| **Zero Cgo** | Pure Go, `CGO_ENABLED=0` | Cannot use C-optimized libraries (BLAS, FAISS, etc.) |

> **Note:** Redis 8+ includes built-in full-text search (BM25) and vector search via the integrated Query Engine. gogaghe is not a Redis replacement — it targets a different architectural niche: an embeddable, in-process cache with cost-aware eviction and character-level search that can be compiled into a single static binary.

---

## ⚡ Key Features

1. **Sub-millisecond Reads/Writes** — all data lives in RAM, no disk I/O on hot paths.
2. **Multi-Resolution Hybrid Search** — three search strategies executed in parallel:
   * **Character N-gram (Surface)**: Trigram-based Sørensen–Dice scoring for typo tolerance and identifier matching.
   * **BM25 (Lexical)**: Term frequency, inverse document frequency, and document length normalization.
   * **Dense Vector (Semantic)**: Cosine similarity over embedding vectors via goroutine pool.
   * **Multi-Stream RRF**: Reciprocal Rank Fusion merging all three ranked lists into a single result without score scale mismatch.
3. **Cost-Aware Eviction** — eviction priority is `CostMs / (AccessCount + 1)`. Items that are cheap to recompute and infrequently accessed are evicted first.
4. **Async Embedding Pipeline** — `Set(auto_embed=true)` returns immediately. Embedding is computed in the background via a buffered goroutine worker pool and HTTP sidecar.
5. **Zero Cgo** — pure Go binary, `CGO_ENABLED=0`, cross-compiles to any `GOOS/GOARCH` target without a C toolchain.
6. **Prometheus Observability** — isolated `prometheus.Registry` with 6 metrics + auto-provisioned Grafana dashboard.

---

## 🏗️ System Architecture

```text
[ Client / Web App ]
         │
         ▼ (HTTP)
[ API Gateway / Backend Service ]
         │
         ├── (1) HybridSearch / Get ──(gRPC)──► [ gogaghe Service ]
         │                                              │
         │                                       Cache Hit?
         │                                       ├── YES → Return (< 1 ms)
         │                                       └── NO  → Cache Miss Signal
         │
         ├── (2) Set(key, val, cost, auto_embed=true) ──► [ gogaghe ]
         │                                                       │
         │                                           Store in RAM (< 1 ms)
         │                                                       │
         │                                        Enqueue background job
         │                                                       ▼
         │                                         [ Sidecar Embedder HTTP ]
         │                                         (Multilingual ONNX model)
         │                                                       │
         │                                         Callback → update Vector
         │
         └── (3) Fallback ──────────────────────► [ PostgreSQL / Primary DB ]
```

### Multi-Tier Hybrid Search Hierarchy

```text
                       HYBRID QUERY
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     SURFACE             LEXICAL             SEMANTIC
   N-gram Index         BM25 Index         Vector Search
 (Character Level)     (Token Level)      (Embedding Level)
   "mirip bentuk?"     "cocok kata?"       "bermakna sama?"
        │                   │                   │
  [doc1, doc2, ...]   [doc1, doc3, ...]   [doc3, doc1, ...]
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
              Reciprocal Rank Fusion (RRF)
                            │
                            ▼
                  Unified Ranked Results
```

**Package dependency (inward only):**

```text
cmd/gogaghe-server/main.go    ← wiring only
  ↓
internal/server/grpc.go       ← gRPC handlers + metrics instrumentation
  ↓
internal/store/engine.go      ← CRUD, TTL, memory tracking (sync.RWMutex + atomic)
  ├── internal/store/ngram.go      ← Character N-gram inverted index (Surface)
  ├── internal/store/bm25.go       ← BM25 inverted index (Lexical)
  ├── internal/store/vector.go     ← Cosine similarity goroutine pool (Semantic)
  ├── internal/store/hybrid.go     ← Multi-stream RRF merger
  └── internal/store/eviction.go   ← Min-heap cost-aware eviction

internal/embedder/client.go   ← async HTTP worker pool → sidecar
                                  (store NEVER imports this)
```

---

## 📋 Feature Specifications

### 1. Core In-Memory Engine (`internal/store/engine.go`)

| Requirement | Detail |
| :--- | :--- |
| Thread Safety | `sync.RWMutex` + `sync/atomic` — zero data races under `-race` |
| Data Model | `Item{Value []byte, CostMs int64, Vector []float32, AccessCount int64, LastAccessedAt, ExpiresAt time.Time}` |
| Memory Limit | Configurable `max_memory_bytes` — `Set()` returns `codes.ResourceExhausted` on overflow |
| TTL Eviction | Background `time.Ticker` goroutine scans expired keys at configurable interval |
| Memory Tracking | Atomic counter updated on every `Set`/`Delete` — lock-free reads on hot path |

### 2. Character N-Gram Surface Search (`internal/store/ngram.go`)

- In-memory character n-gram inverted index (default trigram $n=3$).
- Scoring: Sørensen–Dice coefficient ($2 \times |Q \cap D| / (|Q| + |D|)$) for scale-invariant surface similarity.
- Handles typos, spelling variations, prefixes, identifiers, and technical SKUs (`kubernets` $\leftrightarrow$ `kubernetes`, `Postgres` $\leftrightarrow$ `PostgreSQL`).
- Incremental $O(\text{tokens})$ addition and removal on `Set`/`Delete`.

### 3. BM25 Lexical Search (`internal/store/bm25.go`)

- Pure-Go in-memory inverted index — no external dependencies.
- Tokenizer: lowercase + split on non-alphanumeric characters.
- BM25 parameters: `k1 = 1.5`, `b = 0.75`.
- Incremental indexing on write; full reconstruction (`Rebuild(items)`) supported.
- `Search(query, topK)` returns `[]ScoredKey` sorted by BM25 score descending.

### 4. Dense Vector Search (`internal/store/vector.go`)

- `CosineSimilarity(a, b []float32) float64` — safely handles zero vectors (returns `0.0`).
- `VectorSearch(queryVec, items, topK)` parallelizes across `GOMAXPROCS` goroutine workers.
- Items with no vector or dimension mismatch are silently skipped.

### 5. Multi-Stream Hybrid Search — Reciprocal Rank Fusion (`internal/store/hybrid.go`)

```
RRF score(doc) = Σ  1 / (k + rank_i(doc))
```

- Multi-stream fusion (`RRFMulti`) merges Character N-gram, BM25, and Vector rank lists into a single unified result.
- Default `k = 60.0` (overridable per-request via `HybridSearchRequest.rrf_k`).
- Items appearing across multiple representation layers compound their score; partial matches in only one stream still surface fairly.

### 6. Cost-Aware Smart Eviction (`internal/store/eviction.go`)

```
priority(item) = CostMs / (AccessCount + 1)
```

- Min-heap (`container/heap`) ordered by priority — **lower priority = evicted first**.
- Semantics: cheap-and-unused data is evicted before expensive-and-frequently-accessed data.
- `Evict(engine, targetBytes)` loops until `MemoryUsageBytes() <= targetBytes`.

### 7. gRPC Interface (`api/proto/gogaghe/v1/gogaghe.proto`)

```protobuf
service GogagheService {
  rpc Set(SetRequest)                   returns (SetResponse);
  rpc Get(GetRequest)                   returns (GetResponse);
  rpc Delete(DeleteRequest)             returns (DeleteResponse);

  // Dedicated single-strategy search RPCs:
  rpc SurfaceSearch(SurfaceSearchRequest) returns (SurfaceSearchResponse); // Character N-gram (Trigram / Typo-tolerant)
  rpc LexicalSearch(LexicalSearchRequest) returns (LexicalSearchResponse); // BM25 Inverted Index (Keyword / TF-IDF)
  rpc VectorSearch(VectorSearchRequest)   returns (VectorSearchResponse);  // Dense Vector (Cosine Similarity)

  // Multi-strategy fused hybrid search RPC:
  rpc HybridSearch(HybridSearchRequest)   returns (HybridSearchResponse);  // RRF fusion across selected or all strategies
}
```

- gRPC Server Reflection always enabled — introspect via Postman, Kreya, or `grpcurl` without the `.proto` file.
- Generated Go code in `pkg/gogaghe/v1/` — never edit manually, regenerate with `make proto`.

### 8. Async Embedding Pipeline (`internal/embedder/client.go`)

- `Set` with `auto_embed=true` returns `success=true` to the client immediately (< 1 ms).
- Embedding runs asynchronously via buffered channel + goroutine worker pool.
- The sidecar receives `POST /embed {"text": "..."}` and returns `{"vector": [...]}`.
- Buffer-full requests are dropped silently with a `slog.Warn` — the caller is never blocked.
- Activated via Docker Compose `--profile ai-bundle`.

### 9. Observability (`internal/server/metrics.go`)

| Metric | Type | Label |
| :--- | :--- | :--- |
| `gogaghe_cache_hits_total` | Counter | — |
| `gogaghe_cache_misses_total` | Counter | — |
| `gogaghe_operation_duration_seconds` | Histogram | `operation` |
| `gogaghe_memory_usage_bytes` | Gauge | — |
| `gogaghe_items_count` | Gauge | — |
| `gogaghe_goroutines_active` | Gauge | — |

All metrics registered on an isolated `prometheus.Registry` (not the default global). Grafana dashboard auto-provisioned via Docker Compose volume mount.

---

## 🗄️ Core Data Model

```go
// internal/store/engine.go
type Item struct {
    Value          []byte    // arbitrary payload
    CostMs         int64     // computation cost — used for eviction scoring
    Vector         []float32 // optional embedding — populated async by sidecar
    AccessCount    int64     // incremented on every Get()
    LastAccessedAt time.Time
    ExpiresAt      time.Time // zero = no expiry (TTL disabled)
}
```

---

## 📡 API Reference

### `Set` — Write a key-value item

| Field | Type | Description |
| :--- | :--- | :--- |
| `key` | `string` | Unique cache key |
| `value` | `bytes` | Arbitrary payload |
| `ttl_ms` | `int64` | Time-to-live in ms; `0` = no expiry |
| `cost_ms` | `int64` | Computation cost (milliseconds) for eviction scoring |
| `vector` | `repeated float` | Pre-computed embedding (optional, skips async embed) |
| `auto_embed` | `bool` | Trigger async embedding from `value` text via sidecar |

### `Get` — Retrieve a cached item

| Field | Type | Description |
| :--- | :--- | :--- |
| `found` | `bool` | `false` on cache miss or expired key |
| `value` | `bytes` | Stored payload |
| `vector` | `repeated float` | Embedding (may be empty if not yet generated) |
| `access_count` | `int64` | Lifetime Get count for this key |

### `SurfaceSearch` — Character N-gram (Trigram) surface similarity search

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | Query text (matches typos, prefixes, codes, SKUs via Sørensen–Dice) |
| `top_k` | `int32` | Maximum results to return |

### `LexicalSearch` — BM25 keyword relevance search

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | Lexical query text (scored via BM25 inverted index) |
| `top_k` | `int32` | Maximum results to return |

### `VectorSearch` — Dense vector cosine similarity search

| Field | Type | Description |
| :--- | :--- | :--- |
| `query_vector` | `repeated float` | Semantic query vector |
| `top_k` | `int32` | Maximum results to return |

### `HybridSearch` — Multi-strategy ranked fusion search

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | Query text (drives Surface and/or Lexical strategies) |
| `query_vector` | `repeated float` | Semantic query vector (drives Semantic strategy) |
| `top_k` | `int32` | Maximum results to return |
| `rrf_k` | `float` | RRF constant; defaults to `60.0` if `<= 0` |
| `strategies` | `repeated SearchStrategy` | Optional: choose exact strategies to fuse (`SURFACE`, `LEXICAL`, `SEMANTIC`). Auto-detects if omitted |

---

## 🔧 Tech Stack

| Concern | Solution |
| :--- | :--- |
| **Language & Runtime** | Go 1.25 (>= 1.22) |
| **RPC Protocol** | gRPC / protobuf (`google.golang.org/grpc`) |
| **API Contract** | `buf` toolchain (`buf lint` + `buf generate`) |
| **Config** | `gopkg.in/yaml.v3` |
| **Observability** | `github.com/prometheus/client_golang` |
| **Concurrency** | `sync.RWMutex`, `sync/atomic`, goroutine pools, `container/heap` |
| **Search — Surface** | Character N-gram Inverted Index (`internal/store/ngram.go`) |
| **Search — Lexical** | Pure-Go BM25 Inverted Index (`internal/store/bm25.go`) |
| **Search — Semantic** | Cosine similarity via goroutine pool (`internal/store/vector.go`) |
| **Search — Hybrid** | Multi-stream Reciprocal Rank Fusion (`internal/store/hybrid.go`) |
| **Containerization** | Multi-stage Dockerfile (`FROM scratch`) + Docker Compose v2 profiles |
| **Testing** | `go test -v -race ./...` |

---

## 📐 Non-Functional Requirements

| Requirement | Target |
| :--- | :--- |
| **Read/Write Latency** | <= 1 ms on localhost / private network |
| **Race Safety** | 100% pass `go test -v -race ./...` — zero race conditions tolerated |
| **Zero Cgo** | `CGO_ENABLED=0` — cross-compilable to any Linux/amd64 without a C toolchain |
| **Containerization** | Full stack runnable via `docker compose up` |
| **Binary Size** | Minimal — `FROM scratch` runtime + `-ldflags="-s -w"` |

---

## 🚀 Getting Started

### Prerequisites

- Go >= 1.22
- Docker + Docker Compose v2
- `buf` CLI (`go install github.com/bufbuild/buf/cmd/buf@latest`)

### Run Locally

```bash
# 1. Regenerate protobuf code (first time only)
make proto

# 2. Build binary
make build

# 3. Run server
./bin/gogaghe-server --config configs/config.yaml
```

The gRPC server starts on `:50051`, Prometheus metrics on `:2112`.

### Run with Docker (Basic Mode)

```bash
docker compose -f deployments/docker-compose/docker-compose.yml up --build -d
```

| Service | URL |
| :--- | :--- |
| gogaghe gRPC | `localhost:50051` |
| Prometheus metrics | `http://localhost:2112/metrics` |
| Prometheus UI | `http://localhost:9090` |
| Grafana | `http://localhost:3000` (admin / admin) |

### Run with AI Embedding Sidecar

```bash
docker compose -f deployments/docker-compose/docker-compose.yml --profile ai-bundle up --build -d
```

Adds the `embedder` sidecar on port `8000`. Enable in config: `embedder.enabled: true`.

### Run Tests

```bash
make test   # go test -v -race ./...
```

---

## 📁 Project Structure

```text
gogaghe/
├── api/proto/gogaghe/v1/gogaghe.proto  # gRPC service definition — source of truth
├── build/package/Dockerfile            # Multi-stage: golang:alpine → scratch
├── cmd/gogaghe-server/main.go          # Wiring only — no business logic
├── configs/config.yaml                 # Runtime configuration
├── deployments/docker-compose/         # docker-compose.yml, prometheus.yml, grafana/
├── internal/
│   ├── config/config.go                # YAML config parser
│   ├── embedder/client.go              # Async HTTP embed worker pool
│   ├── server/
│   │   ├── grpc.go                     # GogagheServiceServer + 5 RPC handlers
│   │   ├── grpc_test.go                # gRPC server integration & typo tolerance tests
│   │   └── metrics.go                  # Prometheus registry + /metrics HTTP server
│   └── store/
│       ├── engine.go                   # Core engine: CRUD, TTL, memory tracking
│       ├── ngram.go                    # Character N-gram inverted index (Surface search)
│       ├── ngram_test.go               # N-gram typo tolerance & multi-stream RRF tests
│       ├── bm25.go                     # BM25 inverted index + tokenizer (Lexical search)
│       ├── vector.go                   # Cosine similarity goroutine pool (Semantic search)
│       ├── hybrid.go                   # Multi-stream RRF merger
│       ├── eviction.go                 # Min-heap cost-aware eviction
│       └── engine_test.go              # All store-layer unit + race tests
├── pkg/gogaghe/v1/                     # ⚠ AUTO-GENERATED — run `make proto`
├── buf.yaml / buf.gen.yaml             # Buf toolchain config
└── Makefile                            # make proto | build | run | test | lint
```

---

## 🗺️ Execution Milestones

| Phase | Scope | Key Deliverables |
| :--- | :--- | :--- |
| **Phase 1** | API Contract & Core Storage | `gogaghe.proto`, `engine.go` (CRUD + TTL), unit tests `-race` |
| **Phase 2** | Multi-Resolution Search & Eviction | `ngram.go` (Surface), `bm25.go` (Lexical), `vector.go` (Semantic), `hybrid.go` (RRFMulti), `eviction.go` (min-heap) |
| **Phase 3** | gRPC Server & Async Embedder | `grpc.go` (5 handlers), `main.go` (graceful shutdown), `embedder/client.go` |
| **Phase 4** | Observability & Docker | `metrics.go`, Dockerfile, `docker-compose.yml` (profiles), Grafana dashboard |

---

## ✅ Acceptance Criteria

- [ ] `go test -v -race ./...` passes — zero failures, zero race conditions.
- [ ] `make build` produces a binary under `bin/` with `CGO_ENABLED=0`.
- [ ] `docker compose up` starts gogaghe, Prometheus, and Grafana without errors.
- [ ] `docker compose --profile ai-bundle up` adds the embedder sidecar correctly.
- [ ] gRPC Server Reflection is queryable: `grpcurl -plaintext localhost:50051 list`.
- [ ] `/metrics` returns valid Prometheus text format on port `2112`.
- [ ] Grafana dashboard at `http://localhost:3000` displays all 6 gogaghe metrics.
- [ ] `Set` with `auto_embed=true` returns `success=true` in < 1 ms.
- [ ] `HybridSearch` returns RRF-ranked results combining Character N-gram, BM25, and Cosine similarity.
- [ ] `Evict()` removes lowest-priority items first (cheap + unused before expensive + accessed).
