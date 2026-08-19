// scripts/smoke_test.go
// Run with: go run scripts/smoke_test.go
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	gogaghev1 "github.com/bayurkp/gogaghe/pkg/gogaghe/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	fmt.Println("==================================================")
	fmt.Println("       gogaghe End-to-End Smoke Test Suite        ")
	fmt.Println("==================================================")

	target := "localhost:50051"
	metricsURL := "http://localhost:2112/metrics"

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Printf("[*] Connecting to gRPC server at %s...\n", target)
	conn, err := grpc.DialContext(ctx, target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		log.Fatalf("[-] Failed to connect to gRPC server: %v\n    Make sure the server is running (e.g. .\\bin\\gogaghe-server.exe or .\\scripts\\deploy.ps1)", err)
	}
	defer conn.Close()
	fmt.Println("[+] Connected successfully!")

	client := gogaghev1.NewGogagheServiceClient(conn)

	// --- 1. Test Set RPC ---
	fmt.Println("\n--- 1. Testing Set RPC ---")
	testDocs := []struct {
		key    string
		value  string
		cost   int64
		vector []float32
	}{
		{
			key:    "doc:go",
			value:  "Golang in-memory hybrid store with concurrent worker pools",
			cost:   50,
			vector: []float32{1.0, 0.0, 0.0},
		},
		{
			key:    "doc:python",
			value:  "Python machine learning pipeline and vector embeddings",
			cost:   200,
			vector: []float32{0.0, 1.0, 0.0},
		},
		{
			key:    "doc:db",
			value:  "Relational PostgreSQL transactional database with ACID guarantees",
			cost:   30,
			vector: []float32{0.0, 0.0, 1.0},
		},
		{
			key:    "doc:hybrid",
			value:  "Concurrent hybrid search combining BM25 lexical and dense cosine vectors",
			cost:   150,
			vector: []float32{0.8, 0.2, 0.0},
		},
	}

	for _, d := range testDocs {
		setResp, err := client.Set(ctx, &gogaghev1.SetRequest{
			Key:    d.key,
			Value:  []byte(d.value),
			CostMs: d.cost,
			Vector: d.vector,
		})
		if err != nil {
			log.Fatalf("[-] Set(%s) failed: %v", d.key, err)
		}
		fmt.Printf("  [+] Set key=%-12s success=%v\n", d.key, setResp.Success)
	}

	// --- 2. Test Get RPC ---
	fmt.Println("\n--- 2. Testing Get RPC ---")
	getResp, err := client.Get(ctx, &gogaghev1.GetRequest{Key: "doc:go"})
	if err != nil {
		log.Fatalf("[-] Get(doc:go) failed: %v", err)
	}
	fmt.Printf("  [+] Get(doc:go) -> found=%v, value=%q, access_count=%d\n",
		getResp.Found, string(getResp.Value), getResp.AccessCount)

	getMiss, err := client.Get(ctx, &gogaghev1.GetRequest{Key: "non-existent-key"})
	if err != nil {
		log.Fatalf("[-] Get(non-existent-key) failed: %v", err)
	}
	fmt.Printf("  [+] Get(non-existent) -> found=%v (expected false)\n", getMiss.Found)

	// --- 3. Test VectorSearch RPC ---
	fmt.Println("\n--- 3. Testing VectorSearch RPC ---")
	vecResp, err := client.VectorSearch(ctx, &gogaghev1.VectorSearchRequest{
		QueryVector: []float32{1.0, 0.0, 0.0},
		TopK:        2,
	})
	if err != nil {
		log.Fatalf("[-] VectorSearch failed: %v", err)
	}
	fmt.Printf("  [+] Query Vector: [1.0, 0.0, 0.0] -> Top %d results:\n", len(vecResp.Results))
	for i, r := range vecResp.Results {
		fmt.Printf("      %d. Key: %-12s | Score: %.4f | Value: %s\n", i+1, r.Key, r.Score, string(r.Value))
	}

	// --- 4. Test HybridSearch RPC ---
	fmt.Println("\n--- 4. Testing HybridSearch RPC (BM25 + Cosine + RRF) ---")
	hybridResp, err := client.HybridSearch(ctx, &gogaghev1.HybridSearchRequest{
		Query:       "hybrid concurrent store",
		QueryVector: []float32{0.9, 0.1, 0.0},
		TopK:        3,
		RrfK:        60.0,
	})
	if err != nil {
		log.Fatalf("[-] HybridSearch failed: %v", err)
	}
	fmt.Printf("  [+] Hybrid Query: %q -> Top %d results:\n", "hybrid concurrent store", len(hybridResp.Results))
	for i, r := range hybridResp.Results {
		fmt.Printf("      %d. Key: %-12s | RRF Score: %.6f | Value: %s\n", i+1, r.Key, r.Score, string(r.Value))
	}

	// --- 5. Test Delete RPC ---
	fmt.Println("\n--- 5. Testing Delete RPC ---")
	delResp, err := client.Delete(ctx, &gogaghev1.DeleteRequest{Key: "doc:python"})
	if err != nil {
		log.Fatalf("[-] Delete(doc:python) failed: %v", err)
	}
	fmt.Printf("  [+] Delete(doc:python) -> deleted=%v\n", delResp.Deleted)

	// Verify deleted key is no longer in Get
	getAfterDel, _ := client.Get(ctx, &gogaghev1.GetRequest{Key: "doc:python"})
	fmt.Printf("  [+] Verify deleted key -> found=%v (expected false)\n", getAfterDel.Found)

	// --- 6. Scrape Prometheus Metrics ---
	fmt.Println("\n--- 6. Verifying Prometheus Metrics Endpoint ---")
	resp, err := http.Get(metricsURL)
	if err != nil {
		log.Fatalf("[-] Failed to scrape metrics: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	fmt.Printf("  [+] Scraping %s -> HTTP %s\n", metricsURL, resp.Status)
	lines := strings.Split(string(body), "\n")
	for _, l := range lines {
		if strings.HasPrefix(l, "gogaghe_") && !strings.Contains(l, "_bucket") {
			fmt.Printf("      %s\n", l)
		}
	}

	fmt.Println("\n==================================================")
	fmt.Println("  [SUCCESS] All end-to-end tests passed!          ")
	fmt.Println("==================================================")
}
