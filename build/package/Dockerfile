# build/package/Dockerfile

# ---- Builder stage ----
FROM golang:1.25-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /gogaghe-server ./cmd/gogaghe-server/...

# ---- Runtime stage ----
FROM scratch

COPY --from=builder /gogaghe-server /gogaghe-server
COPY --from=builder /app/configs/config.yaml /configs/config.yaml

EXPOSE 50051 2112

ENTRYPOINT ["/gogaghe-server", "--config", "/configs/config.yaml"]
