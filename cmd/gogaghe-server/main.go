// cmd/gogaghe-server/main.go
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"github.com/bayurkp/gogaghe/internal/config"
	"github.com/bayurkp/gogaghe/internal/embedder"
	"github.com/bayurkp/gogaghe/internal/server"
	"github.com/bayurkp/gogaghe/internal/store"
	gogaghev1 "github.com/bayurkp/gogaghe/pkg/gogaghe/v1"
)

func main() {
	configPath := flag.String("config", "configs/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "err", err)
		os.Exit(1)
	}

	// --- Core engine ---
	engine := store.NewEngine(
		cfg.Store.MaxMemoryBytes,
		time.Duration(cfg.Store.TTLCheckIntervalSeconds)*time.Second,
	)
	bm25 := store.NewBM25IndexWithParams(cfg.Search.Lexical.BM25K1, cfg.Search.Lexical.BM25B)
	ngram := store.NewNgramIndex(cfg.Search.Surface.NgramSize)
	metrics := server.NewMetrics()

	// --- Embedder sidecar client (optional) ---
	var emb *embedder.Client
	if cfg.Embedder.Enabled {
		emb = embedder.NewClientWithCallback(cfg.Embedder, func(key string, vector []float32) {
			item, ok := engine.Get(key)
			if !ok {
				return
			}
			item.Vector = vector
			if err := engine.Set(key, item); err != nil {
				slog.Warn("embedder callback: could not update vector", "key", key, "err", err)
			}
			itemsSnapshot := engine.Items()
			bm25.Rebuild(itemsSnapshot)
			ngram.Rebuild(itemsSnapshot)
		})
	}

	// --- gRPC server ---
	grpcSrv := grpc.NewServer()
	gogaghev1.RegisterGogagheServiceServer(grpcSrv, server.NewGogagheServerWithConfig(engine, bm25, ngram, metrics, emb, cfg.Search))
	reflection.Register(grpcSrv)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Server.GRPCPort))
	if err != nil {
		slog.Error("failed to listen", "err", err)
		os.Exit(1)
	}

	// --- Metrics HTTP server (non-blocking) ---
	go func() {
		slog.Info("metrics server starting", "port", cfg.Server.MetricsPort)
		if err := metrics.StartHTTPServer(cfg.Server.MetricsPort); err != nil {
			slog.Error("metrics server error", "err", err)
		}
	}()

	// --- Graceful shutdown ---
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		slog.Info("gRPC server starting", "port", cfg.Server.GRPCPort)
		if err := grpcSrv.Serve(lis); err != nil {
			slog.Error("gRPC serve error", "err", err)
		}
	}()

	<-quit
	slog.Info("shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = ctx

	grpcSrv.GracefulStop()
	engine.Stop()
	if emb != nil {
		emb.Stop()
	}
	slog.Info("shutdown complete")
}
