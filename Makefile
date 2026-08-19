.PHONY: all proto build run test lint clean docker-up docker-up-ai docker-down

BINARY_NAME=gogaghe-server
BUILD_DIR=./bin

all: proto build

proto:
	buf generate

build:
	CGO_ENABLED=0 go build -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/gogaghe-server/...

run: build
	$(BUILD_DIR)/$(BINARY_NAME)

test:
	go test -v -race ./...

lint:
	buf lint
	go vet ./...

clean:
	rm -rf $(BUILD_DIR)

docker-up:
	docker compose -f deployments/docker-compose/docker-compose.yml up --build -d

docker-up-ai:
	docker compose -f deployments/docker-compose/docker-compose.yml --profile ai-bundle up --build -d

docker-down:
	docker compose -f deployments/docker-compose/docker-compose.yml down
