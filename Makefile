# fastembed-go Makefile
#
# Simplifies running tests with proper ONNX Runtime configuration

# ONNX Runtime configuration
ONNX_VERSION := 1.22.0
ONNX_DIR := $(shell cd .. && pwd)/onnxruntime
ONNX_LINUX_GPU := onnxruntime-linux-x64-gpu-$(ONNX_VERSION)
ONNX_LINUX_CPU := onnxruntime-linux-x64-$(ONNX_VERSION)
ONNX_DARWIN := onnxruntime-osx-arm64-$(ONNX_VERSION)

# Auto-detect platform and set ONNX_PATH
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
    # Prefer GPU version if available
    ifneq ($(wildcard $(ONNX_DIR)/$(ONNX_LINUX_GPU)/lib/libonnxruntime.so),)
        ONNX_PATH := $(ONNX_DIR)/$(ONNX_LINUX_GPU)/lib/libonnxruntime.so
    else ifneq ($(wildcard $(ONNX_DIR)/$(ONNX_LINUX_CPU)/lib/libonnxruntime.so),)
        ONNX_PATH := $(ONNX_DIR)/$(ONNX_LINUX_CPU)/lib/libonnxruntime.so
    endif
endif

ifeq ($(UNAME_S),Darwin)
    ifneq ($(wildcard $(ONNX_DIR)/$(ONNX_DARWIN)/lib/libonnxruntime.dylib),)
        ONNX_PATH := $(ONNX_DIR)/$(ONNX_DARWIN)/lib/libonnxruntime.dylib
    endif
endif

# Allow override via environment
ifdef ONNX_PATH_OVERRIDE
    ONNX_PATH := $(ONNX_PATH_OVERRIDE)
endif

# Export for subprocesses
export ONNX_PATH

.PHONY: help test test-short test-cuda test-all test-verbose build clean download-onnx lint fmt check

help: ## Show this help
	@echo "fastembed-go Makefile"
	@echo ""
	@echo "ONNX_PATH: $(ONNX_PATH)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: check-onnx ## Run tests (skip large models and CUDA)
	go test -short -v ./...

test-verbose: check-onnx ## Run all tests with verbose output (includes large model downloads)
	go test -v ./...

test-cuda: check-onnx ## Run CUDA tests (requires NVIDIA GPU)
	TEST_CUDA=1 go test -v -run TestCUDAExecution ./...

test-all: check-onnx ## Run all tests including CUDA
	TEST_CUDA=1 go test -v ./...

test-canonical: check-onnx ## Run canonical values test only
	go test -v -run TestCanonicalValues ./...

test-e5: check-onnx ## Run E5 model tests
	go test -v -run "TestMultilingualE5|TestInstructEmbed" ./...

test-gemma: check-onnx ## Run Gemma model test
	go test -v -run TestEmbeddingGemma ./...

test-pooling: check-onnx ## Run pooling test
	go test -v -run TestPoolingOverride ./...

build: ## Build the package
	go build ./...

lint: ## Run golangci-lint
	golangci-lint run ./...

fmt: ## Format code
	go fmt ./...
	gofumpt -w .

check: build lint ## Run build and lint checks

clean: ## Clean test cache and local_cache
	go clean -testcache
	rm -rf local_cache

clean-models: ## Remove downloaded models (keeps ONNX runtime)
	rm -rf local_cache

check-onnx:
	@if [ -z "$(ONNX_PATH)" ]; then \
		echo "Error: ONNX_PATH not set and could not auto-detect ONNX Runtime"; \
		echo ""; \
		echo "Options:"; \
		echo "  1. Run 'make download-onnx' to download ONNX Runtime"; \
		echo "  2. Set ONNX_PATH_OVERRIDE=/path/to/libonnxruntime.so"; \
		echo ""; \
		exit 1; \
	fi
	@if [ ! -f "$(ONNX_PATH)" ]; then \
		echo "Error: ONNX Runtime not found at $(ONNX_PATH)"; \
		echo "Run 'make download-onnx' to download it"; \
		exit 1; \
	fi
	@echo "Using ONNX Runtime: $(ONNX_PATH)"

download-onnx: ## Download ONNX Runtime for your platform
	@mkdir -p $(ONNX_DIR)
ifeq ($(UNAME_S),Linux)
	@echo "Downloading ONNX Runtime $(ONNX_VERSION) for Linux (GPU)..."
	@cd $(ONNX_DIR) && \
		curl -L -O "https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_VERSION)/$(ONNX_LINUX_GPU).tgz" && \
		tar -xzf $(ONNX_LINUX_GPU).tgz && \
		rm $(ONNX_LINUX_GPU).tgz
	@echo "Downloaded to $(ONNX_DIR)/$(ONNX_LINUX_GPU)"
endif
ifeq ($(UNAME_S),Darwin)
	@echo "Downloading ONNX Runtime $(ONNX_VERSION) for macOS (ARM64)..."
	@cd $(ONNX_DIR) && \
		curl -L -O "https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_VERSION)/$(ONNX_DARWIN).tgz" && \
		tar -xzf $(ONNX_DARWIN).tgz && \
		rm $(ONNX_DARWIN).tgz
	@echo "Downloaded to $(ONNX_DIR)/$(ONNX_DARWIN)"
endif
	@echo ""
	@echo "ONNX Runtime installed. You can now run 'make test'"

# Docker targets
DOCKER_IMAGE := fastembed-go-test

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) ..

docker-test: docker-build ## Run tests in Docker (downloads models fresh)
	docker run --rm $(DOCKER_IMAGE)

docker-test-cached: docker-build ## Run tests with cached models (mount local_cache)
	docker run --rm -v $(PWD)/local_cache:/src/fastembed-go/local_cache $(DOCKER_IMAGE)

docker-test-gpu: docker-build ## Run tests with GPU support
	docker run --rm --gpus all -e TEST_CUDA=1 \
		-v $(PWD)/local_cache:/src/fastembed-go/local_cache \
		$(DOCKER_IMAGE) go test -v ./...

docker-test-all-gpu: docker-build ## Run all tests with GPU (includes CUDA)
	docker run --rm --gpus all -e TEST_CUDA=1 \
		-v $(PWD)/local_cache:/src/fastembed-go/local_cache \
		$(DOCKER_IMAGE) go test -v ./...

docker-shell: docker-build ## Open shell in Docker container
	docker run --rm -it -v $(PWD)/local_cache:/src/fastembed-go/local_cache $(DOCKER_IMAGE) /bin/bash

# Shortcuts
t: test
tv: test-verbose
tc: test-cuda
ta: test-all
dt: docker-test
dtc: docker-test-cached
dtg: docker-test-gpu
