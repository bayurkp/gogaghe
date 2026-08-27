// internal/store/lsa.go
package store

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

const (
	defaultLSADim = 64
	svdPowerIters = 4
)

// LSAIndex is an in-memory Latent Semantic Analysis index using pure-Go Truncated SVD.
type LSAIndex struct {
	mu          sync.RWMutex
	dimK        int
	docKeys     []string               // doc index -> key
	docKeyToIdx map[string]int         // key -> doc index
	terms       []string               // term index -> token
	termToIdx   map[string]int         // token -> term index
	termIDF     []float64              // term index -> idf
	uMatrix     [][]float64            // M x K (term projection)
	sigmaInv    []float64              // 1 x K (1 / singular values)
	docVectors  [][]float64            // N x K (low-rank doc embeddings, L2 normalized)
	tfidfIndex  *TFIDFIndex            // underlying TF-IDF representation
}

// NewLSAIndex creates an LSAIndex with default rank k (64).
func NewLSAIndex() *LSAIndex {
	return NewLSAIndexWithDim(defaultLSADim)
}

// NewLSAIndexWithDim creates an LSAIndex with custom rank k.
func NewLSAIndexWithDim(dimK int) *LSAIndex {
	if dimK <= 0 {
		dimK = defaultLSADim
	}
	return &LSAIndex{
		dimK:        dimK,
		docKeyToIdx: make(map[string]int),
		termToIdx:   make(map[string]int),
		tfidfIndex:  NewTFIDFIndex(),
	}
}

// Rebuild builds the full TF-IDF matrix and computes Truncated SVD.
func (l *LSAIndex) Rebuild(items map[string]Item) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.tfidfIndex.Rebuild(items)
	if len(items) == 0 {
		l.reset()
		return
	}

	// 1. Build vocabulary and doc list
	l.docKeys = make([]string, 0, len(items))
	l.docKeyToIdx = make(map[string]int, len(items))
	l.terms = make([]string, 0, len(l.tfidfIndex.invertedIndex))
	l.termToIdx = make(map[string]int, len(l.tfidfIndex.invertedIndex))

	for key := range items {
		l.docKeyToIdx[key] = len(l.docKeys)
		l.docKeys = append(l.docKeys, key)
	}

	for tok := range l.tfidfIndex.invertedIndex {
		l.termToIdx[tok] = len(l.terms)
		l.terms = append(l.terms, tok)
	}

	m := len(l.terms)
	n := len(l.docKeys)
	if m == 0 || n == 0 {
		l.reset()
		return
	}

	// Effective dimension cannot exceed min(M, N)
	k := l.dimK
	if k > m {
		k = m
	}
	if k > n {
		k = n
	}

	// 2. Precompute IDFs
	l.termIDF = make([]float64, m)
	for i, tok := range l.terms {
		df := float64(len(l.tfidfIndex.invertedIndex[tok]))
		l.termIDF[i] = math.Log((float64(n)-df+0.5)/(df+0.5) + 1)
	}

	// 3. Build sparse term-document matrix A (M x N)
	// matrixA[termIdx][docIdx] = tfidf
	matrixA := make([][]float64, m)
	for i := range matrixA {
		matrixA[i] = make([]float64, n)
	}

	for termIdx, tok := range l.terms {
		postings := l.tfidfIndex.invertedIndex[tok]
		idf := l.termIDF[termIdx]
		for docKey, count := range postings {
			docIdx, ok := l.docKeyToIdx[docKey]
			if !ok {
				continue
			}
			tfWeight := 1.0 + math.Log(float64(count))
			matrixA[termIdx][docIdx] = tfWeight * idf
		}
	}

	// 4. Compute Truncated SVD via Randomized SVD / Power Iteration
	u, s, v := computeTruncatedSVD(matrixA, m, n, k, svdPowerIters)

	l.uMatrix = u
	l.sigmaInv = make([]float64, k)
	for i := 0; i < k; i++ {
		if s[i] > 1e-12 {
			l.sigmaInv[i] = 1.0 / s[i]
		} else {
			l.sigmaInv[i] = 0
		}
	}

	// 5. Document embeddings: D_k = V * Sigma (N x K), normalized to unit length
	l.docVectors = make([][]float64, n)
	for docIdx := 0; docIdx < n; docIdx++ {
		vec := make([]float64, k)
		var normSq float64
		for j := 0; j < k; j++ {
			val := v[docIdx][j] * s[j]
			vec[j] = val
			normSq += val * val
		}
		norm := math.Sqrt(normSq)
		if norm > 0 {
			for j := 0; j < k; j++ {
				vec[j] /= norm
			}
		}
		l.docVectors[docIdx] = vec
	}
}

func (l *LSAIndex) reset() {
	l.docKeys = nil
	l.docKeyToIdx = make(map[string]int)
	l.terms = nil
	l.termToIdx = make(map[string]int)
	l.termIDF = nil
	l.uMatrix = nil
	l.sigmaInv = nil
	l.docVectors = nil
}

// Search projects the query into latent space and returns top-k documents scored by Cosine Similarity.
func (l *LSAIndex) Search(query string, topK int) []ScoredKey {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if len(l.docVectors) == 0 || len(l.uMatrix) == 0 {
		return nil
	}

	qTokens := tokenize(query)
	if len(qTokens) == 0 {
		return nil
	}

	qTfMap := make(map[string]int)
	for _, tok := range qTokens {
		qTfMap[tok]++
	}

	k := len(l.sigmaInv)
	m := len(l.terms)
	n := len(l.docKeys)

	// Build query TF-IDF vector q (1 x M)
	qVec := make([]float64, m)
	for tok, count := range qTfMap {
		termIdx, ok := l.termToIdx[tok]
		if !ok {
			continue
		}
		tfWeight := 1.0 + math.Log(float64(count))
		qVec[termIdx] = tfWeight * l.termIDF[termIdx]
	}

	// Project query to latent space: q_lsa = (q * U) * SigmaInv (1 x K)
	qLSA := make([]float64, k)
	var qNormSq float64
	for j := 0; j < k; j++ {
		var dot float64
		for i := 0; i < m; i++ {
			if qVec[i] != 0 {
				dot += qVec[i] * l.uMatrix[i][j]
			}
		}
		val := dot * l.sigmaInv[j]
		qLSA[j] = val
		qNormSq += val * val
	}

	qNorm := math.Sqrt(qNormSq)
	if qNorm == 0 {
		return nil
	}
	for j := 0; j < k; j++ {
		qLSA[j] /= qNorm
	}

	// Compute Cosine Similarity with all pre-normalized document vectors
	ranked := make([]ScoredKey, 0, n)
	for docIdx, docVec := range l.docVectors {
		var cosine float64
		for j := 0; j < k; j++ {
			cosine += qLSA[j] * docVec[j]
		}
		if cosine > 0 {
			ranked = append(ranked, ScoredKey{
				Key:   l.docKeys[docIdx],
				Score: cosine,
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

// computeTruncatedSVD computes rank-k SVD: A ≈ U * Sigma * V^T using Randomized SVD.
// matrixA is M x N. Returns U (M x k), S (k), V (N x k).
func computeTruncatedSVD(a [][]float64, m, n, k, powerIters int) (u [][]float64, s []float64, v [][]float64) {
	// 1. Generate random Gaussian matrix Omega (N x k)
	rng := rand.New(rand.NewSource(42))
	omega := make([][]float64, n)
	for i := range omega {
		omega[i] = make([]float64, k)
		for j := range omega[i] {
			omega[i][j] = rng.NormFloat64()
		}
	}

	// 2. Y = A * Omega (M x k)
	y := matMul(a, omega, m, n, k)

	// Power iterations to amplify top singular values: Y = (A * A^T)^p * (A * Omega)
	for iter := 0; iter < powerIters; iter++ {
		// Q, _ = QR(Y)
		q := qrDecomp(y, m, k)
		// Y = A * (A^T * Q)
		atQ := matMulAT(a, q, m, n, k) // N x k
		y = matMul(a, atQ, m, n, k)     // M x k
	}

	// 3. Orthonormalize Y to obtain basis Q (M x k)
	q := qrDecomp(y, m, k)

	// 4. Form smaller matrix B = Q^T * A (k x N)
	b := matMulQT(q, a, m, k, n) // k x N

	// 5. Compute Gram matrix G = B * B^T (k x k)
	g := make([][]float64, k)
	for i := range g {
		g[i] = make([]float64, k)
		for j := range g[i] {
			var sum float64
			for col := 0; col < n; col++ {
				sum += b[i][col] * b[j][col]
			}
			g[i][j] = sum
		}
	}

	// 6. Eigen-decomposition of small k x k matrix G via Jacobi eigenvalue algorithm
	eigVals, eigVecs := jacobiEigen(g, k)

	// Singular values of B: s_i = sqrt(max(0, eigVal_i))
	s = make([]float64, k)
	for i := 0; i < k; i++ {
		if eigVals[i] > 0 {
			s[i] = math.Sqrt(eigVals[i])
		}
	}

	// U = Q * eigVecs (M x k)
	u = matMul(q, eigVecs, m, k, k)

	// V = B^T * eigVecs * diag(1/s) (N x k)
	v = make([][]float64, n)
	for i := range v {
		v[i] = make([]float64, k)
	}

	for doc := 0; doc < n; doc++ {
		for col := 0; col < k; col++ {
			if s[col] > 1e-12 {
				var sum float64
				for row := 0; row < k; row++ {
					sum += b[row][doc] * eigVecs[row][col]
				}
				v[doc][col] = sum / s[col]
			}
		}
	}

	return u, s, v
}

func matMul(a, b [][]float64, rowsA, colsA, colsB int) [][]float64 {
	res := make([][]float64, rowsA)
	for i := range res {
		res[i] = make([]float64, colsB)
		for k := 0; k < colsA; k++ {
			aVal := a[i][k]
			if aVal != 0 {
				for j := 0; j < colsB; j++ {
					res[i][j] += aVal * b[k][j]
				}
			}
		}
	}
	return res
}

func matMulAT(a, q [][]float64, m, n, k int) [][]float64 {
	// A^T (N x M) * Q (M x k) -> (N x k)
	res := make([][]float64, n)
	for i := range res {
		res[i] = make([]float64, k)
	}
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			aTC := a[r][c] // A[r][c] is (A^T)[c][r]
			if aTC != 0 {
				for j := 0; j < k; j++ {
					res[c][j] += aTC * q[r][j]
				}
			}
		}
	}
	return res
}

func matMulQT(q, a [][]float64, m, k, n int) [][]float64 {
	// Q^T (k x M) * A (M x N) -> (k x N)
	res := make([][]float64, k)
	for i := range res {
		res[i] = make([]float64, n)
	}
	for i := 0; i < k; i++ {
		for row := 0; row < m; row++ {
			qVal := q[row][i]
			if qVal != 0 {
				for col := 0; col < n; col++ {
					res[i][col] += qVal * a[row][col]
				}
			}
		}
	}
	return res
}

// qrDecomp computes thin QR decomposition using modified Gram-Schmidt.
func qrDecomp(a [][]float64, rows, cols int) [][]float64 {
	q := make([][]float64, rows)
	for i := range q {
		q[i] = make([]float64, cols)
		copy(q[i], a[i])
	}

	for j := 0; j < cols; j++ {
		var normSq float64
		for i := 0; i < rows; i++ {
			normSq += q[i][j] * q[i][j]
		}
		norm := math.Sqrt(normSq)
		if norm > 1e-12 {
			for i := 0; i < rows; i++ {
				q[i][j] /= norm
			}
		}
		for k := j + 1; k < cols; k++ {
			var dot float64
			for i := 0; i < rows; i++ {
				dot += q[i][j] * q[i][k]
			}
			for i := 0; i < rows; i++ {
				q[i][k] -= dot * q[i][j]
			}
		}
	}
	return q
}

// jacobiEigen computes eigenvalues and eigenvectors of a symmetric k x k matrix.
func jacobiEigen(matrix [][]float64, k int) ([]float64, [][]float64) {
	a := make([][]float64, k)
	v := make([][]float64, k)
	for i := range a {
		a[i] = make([]float64, k)
		copy(a[i], matrix[i])
		v[i] = make([]float64, k)
		v[i][i] = 1.0 // Identity matrix
	}

	const maxIters = 50
	for iter := 0; iter < maxIters; iter++ {
		// Find largest off-diagonal element
		var maxVal float64
		p, q := 0, 1
		for i := 0; i < k; i++ {
			for j := i + 1; j < k; j++ {
				absVal := math.Abs(a[i][j])
				if absVal > maxVal {
					maxVal = absVal
					p, q = i, j
				}
			}
		}
		if maxVal < 1e-12 {
			break
		}

		app := a[p][p]
		aqq := a[q][q]
		apq := a[p][q]

		theta := 0.5 * math.Atan2(2.0*apq, aqq-app)
		c := math.Cos(theta)
		s := math.Sin(theta)

		// Update A
		a[p][p] = c*c*app - 2.0*s*c*apq + s*s*aqq
		a[q][q] = s*s*app + 2.0*s*c*apq + c*c*aqq
		a[p][q] = 0.0
		a[q][p] = 0.0

		for i := 0; i < k; i++ {
			if i != p && i != q {
				aip := a[i][p]
				aiq := a[i][q]
				a[i][p] = c*aip - s*aiq
				a[p][i] = a[i][p]
				a[i][q] = s*aip + c*aiq
				a[q][i] = a[i][q]
			}
		}

		// Update V
		for i := 0; i < k; i++ {
			vip := v[i][p]
			viq := v[i][q]
			v[i][p] = c*vip - s*viq
			v[i][q] = s*vip + c*viq
		}
	}

	eigVals := make([]float64, k)
	for i := 0; i < k; i++ {
		eigVals[i] = a[i][i]
	}

	// Sort eigenvalues and eigenvectors descending
	type pair struct {
		val float64
		vec []float64
	}
	pairs := make([]pair, k)
	for i := 0; i < k; i++ {
		vec := make([]float64, k)
		for row := 0; row < k; row++ {
			vec[row] = v[row][i]
		}
		pairs[i] = pair{val: eigVals[i], vec: vec}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val
	})

	sortedVals := make([]float64, k)
	sortedVecs := make([][]float64, k)
	for row := 0; row < k; row++ {
		sortedVecs[row] = make([]float64, k)
	}
	for i := 0; i < k; i++ {
		sortedVals[i] = pairs[i].val
		for row := 0; row < k; row++ {
			sortedVecs[row][i] = pairs[i].vec[row]
		}
	}

	return sortedVals, sortedVecs
}
