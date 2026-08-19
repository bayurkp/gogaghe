#!/usr/bin/env bash
set -euo pipefail

# gogaghe Linux/macOS/WSL deployment script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/deployments/docker-compose/docker-compose.yml"
CONFIG_FILE="${PROJECT_ROOT}/configs/config.yaml"

ACTION="${1:-up}"

echo "=========================================="
echo "  gogaghe Deployment & Management Script  "
echo "=========================================="

case "${ACTION}" in
    up)
        echo "[*] Deploying standard Docker Compose stack..."
        docker compose -f "${COMPOSE_FILE}" up --build -d
        echo "[+] gogaghe is live!"
        echo "    gRPC Server   : localhost:50051"
        echo "    Prometheus    : http://localhost:9090"
        echo "    Grafana       : http://localhost:3000 (admin/admin)"
        echo "    Metrics       : http://localhost:2112/metrics"
        ;;
    up-ai)
        echo "[*] Deploying Docker Compose stack with AI Embedding Sidecar..."
        docker compose -f "${COMPOSE_FILE}" --profile ai-bundle up --build -d
        echo "[+] gogaghe AI-bundle is live!"
        echo "    Sidecar Embedder: http://localhost:8000"
        ;;
    down)
        echo "[*] Stopping Docker Compose stack..."
        docker compose -f "${COMPOSE_FILE}" down
        echo "[+] All services stopped."
        ;;
    local)
        echo "[*] Building local binary (CGO_ENABLED=0)..."
        CGO_ENABLED=0 go build -ldflags="-s -w" -o "${PROJECT_ROOT}/bin/gogaghe-server" "${PROJECT_ROOT}/cmd/gogaghe-server/..."
        echo "[*] Running gogaghe-server locally..."
        "${PROJECT_ROOT}/bin/gogaghe-server" --config "${CONFIG_FILE}"
        ;;
    status)
        docker compose -f "${COMPOSE_FILE}" ps
        curl -s http://localhost:2112/metrics | head -n 15 || echo "[-] Server not responding."
        ;;
    *)
        echo "Usage: $0 {up|up-ai|down|local|status}"
        exit 1
        ;;
esac
