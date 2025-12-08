<div align="center">
 <h1 style="display: inline-block; vertical-align: middle;">
    <a href="https://crates.io/crates/fastembed">FastEmbed-go</a>
    <img src="https://github.com/Anush008/fastembed-rs/assets/46051506/4bd3cefe-12da-48b9-8cc2-7489145c9cb5" style="display: inline-block; vertical-align: middle; width: auto; height: 100px;">
 </h1>
 <h3>Go implementation of <a href="https://github.com/qdrant/fastembed" target="_blank">@Qdrant/fastembed</a></h3>
  <a href="https://pkg.go.dev/github.com/anush008/fastembed-go"><img src="https://pkg.go.dev/badge/github.com/anush008/fastembed-go.svg" alt="Go Reference"></a>
  <a href="https://github.com/Anush008/fastembed-go/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-mit-blue.svg" alt="MIT Licensed"></a>
  <a href="https://github.com/Anush008/fastembed-go/actions/workflows/release.yml"><img src="https://github.com/Anush008/fastembed-go/actions/workflows/release.yml/badge.svg?branch=main" alt="Semantic release"></a>
</div>

## Features

- Generate text embeddings locally using ONNX Runtime
- 10 pre-trained models including multilingual support
- CUDA GPU acceleration support
- Configurable pooling strategies (CLS / Mean)
- Batch embeddings with parallelism using goroutines
- Automatic model downloading from HuggingFace

## Not looking for Go?

- Python: [fastembed](https://github.com/qdrant/fastembed)
- Rust: [fastembed-rs](https://github.com/Anush008/fastembed-rs)
- JavaScript: [fastembed-js](https://github.com/Anush008/fastembed-js)

## Requirements

- Go 1.25+
- ONNX Runtime 1.23.0
- (Optional) CUDA 12.x + cuDNN 9 for GPU acceleration

## Quick Start

### 1. Install ONNX Runtime

**Option A: Using Make (recommended)**

```bash
make download-onnx
```

**Option B: Manual download**

Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0):

```bash
# Linux (GPU)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-gpu-1.23.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.23.0.tgz

# macOS (ARM64)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-osx-arm64-1.23.0.tgz
tar -xzf onnxruntime-osx-arm64-1.23.0.tgz
```

### 2. Set Environment Variable

```bash
# Linux
export ONNX_PATH="/path/to/onnxruntime-linux-x64-gpu-1.23.0/lib/libonnxruntime.so"

# macOS
export ONNX_PATH="/path/to/onnxruntime-osx-arm64-1.23.0/lib/libonnxruntime.dylib"
```

> **Note:** The Makefile auto-detects ONNX Runtime in `../onnxruntime/`, so you can skip this step if using `make test`.

### 3. Install the Package

```bash
go get -u github.com/anush008/fastembed-go
```

### 4. Generate Embeddings

```go
package main

import (
    "fmt"
    "log"

    fastembed "github.com/anush008/fastembed-go"
)

func main() {
    // Initialize with default model (BGESmallENV15)
    fe, err := fastembed.NewFlagEmbedding(nil)
    if err != nil {
        log.Fatal(err)
    }
    defer fe.Destroy()

    // Generate embeddings
    texts := []string{
        "Hello, world!",
        "This is a test sentence.",
    }

    embeddings, err := fe.Embed(texts, 256) // batch size 256
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Generated %d embeddings of dimension %d\n",
        len(embeddings), len(embeddings[0]))
}
```

## Available Models

| Model | Dimensions | Pooling | Description |
|-------|------------|---------|-------------|
| `BGESmallENV15` (default) | 384 | CLS | Fast English model |
| `BGEBaseENV15` | 768 | CLS | Base English model |
| `BGESmallEN` | 384 | CLS | Small English model |
| `BGEBaseEN` | 768 | CLS | Base English model (v1) |
| `BGESmallZH` | 512 | CLS | Chinese model |
| `AllMiniLML6V2` | 384 | CLS | MiniLM sentence transformer |
| `MultilingualE5Large` | 1024 | Mean | 100+ language support |
| `MultilingualE5LargeInstruct` | 1024 | Mean | Task-specific with instructions |
| `EmbeddingGemma300M` | 768 | Mean | Google Gemma (FP32) |
| `EmbeddingGemma300MQ4` | 768 | Mean | Google Gemma (Q4 quantized) |

## Usage Examples

### Choose a Specific Model

```go
fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
    Model: fastembed.MultilingualE5Large,
})
```

### Semantic Search (Query/Passage)

```go
// Embed query with "query: " prefix
queryEmb, err := fe.QueryEmbed("What is machine learning?")

// Embed passages with "passage: " prefix
passages := []string{
    "Machine learning is a subset of AI...",
    "The weather today is sunny...",
}
passageEmbs, err := fe.PassageEmbed(passages, 256)
```

### Instruction-Based Embeddings (E5-Instruct)

```go
fe, _ := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
    Model: fastembed.MultilingualE5LargeInstruct,
})

task := "Given a query, retrieve relevant passages"
texts := []string{"What is machine learning?", "How do neural networks work?"}

embeddings, err := fe.InstructEmbed(texts, task, 256)
```

### Custom Pooling Strategy

```go
// Override default pooling (CLS vs Mean)
meanPooling := fastembed.PoolingMean
fe, _ := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
    Model:   fastembed.BGESmallENV15,
    Pooling: &meanPooling,
})
```

### CUDA GPU Acceleration

```go
fe, _ := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
    Model:        fastembed.BGESmallENV15,
    UseCUDA:      true,
    CUDADeviceID: 0, // GPU device index
})
```

### Custom Cache Directory

```go
fe, _ := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
    Model:    fastembed.BGESmallENV15,
    CacheDir: "/custom/model/cache",
})
```

### List Supported Models

```go
models := fastembed.ListSupportedModels()
for _, m := range models {
    fmt.Printf("%s (%d dims): %s\n", m.Model, m.Dim, m.Description)
}
```

## API Reference

### InitOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Model` | `EmbeddingModel` | `BGESmallENV15` | Model to use |
| `CacheDir` | `string` | `local_cache` | Directory for downloaded models |
| `MaxLength` | `int` | `512` | Maximum token sequence length |
| `ShowDownloadProgress` | `*bool` | `true` | Show download progress bar |
| `Pooling` | `*Pooling` | model default | Override pooling strategy |
| `UseCUDA` | `bool` | `false` | Enable CUDA GPU acceleration |
| `CUDADeviceID` | `int` | `0` | GPU device index |

### Methods

| Method | Description |
|--------|-------------|
| `Embed(texts, batchSize)` | Generate embeddings for multiple texts |
| `QueryEmbed(text)` | Embed single text with "query: " prefix |
| `PassageEmbed(texts, batchSize)` | Embed texts with "passage: " prefix |
| `InstructEmbed(texts, task, batchSize)` | Embed with instruction prefix (E5-Instruct) |
| `InstructQueryEmbed(query, task)` | Single query with instruction |
| `Destroy()` | Clean up resources (must call when done) |

### Pooling Strategies

| Strategy | Description |
|----------|-------------|
| `PoolingCls` | Extract CLS token embedding (first token) |
| `PoolingMean` | Mean of all token embeddings weighted by attention mask |

## Development

### Run Tests

```bash
# Quick tests (skips large model downloads)
make test

# Full tests with all models
make test-verbose

# CUDA GPU tests
make test-cuda

# All tests including CUDA
make test-all

# Specific test suites
make test-canonical  # Original model validation
make test-e5         # E5 model tests
make test-gemma      # Gemma model test
```

### Other Commands

```bash
make help           # Show all commands
make build          # Build check
make lint           # Run golangci-lint
make fmt            # Format code
make clean          # Clean test cache and models
make download-onnx  # Download ONNX Runtime
```

### Docker

Run tests in a container with CUDA support pre-configured:

```bash
# Build the Docker image
make docker-build

# Run tests (downloads models fresh each time)
make docker-test

# Run tests with cached models (faster for repeated runs)
make docker-test-cached

# Run tests with GPU support
make docker-test-gpu

# Open an interactive shell in the container
make docker-shell
```

The Docker image includes:

- CUDA 12.8 + cuDNN 9
- Go 1.23
- ONNX Runtime 1.23.0 (GPU)

Mount your local model cache for faster repeated runs:

```bash
docker run --rm -v $(pwd)/local_cache:/src/fastembed-go/local_cache fastembed-go-test
```

## CUDA Setup

For GPU acceleration:

1. **CUDA Toolkit 12.x**: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. **cuDNN 9**: [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn)
3. **ONNX Runtime GPU**: Included in `make download-onnx` for Linux

Verify CUDA works:

```bash
make test-cuda
```

## Troubleshooting

### "Platform-specific initialization failed: Error loading ONNX shared library"

Set the `ONNX_PATH` environment variable:

```bash
export ONNX_PATH="/path/to/libonnxruntime.so"
```

Or use the Makefile which auto-detects it:

```bash
make test
```

### "CUDA provider not available"

Ensure you have:

- ONNX Runtime GPU version (not CPU-only)
- CUDA 12.x installed and in PATH
- cuDNN 9 installed
- Compatible NVIDIA drivers

### Model download fails

Check your internet connection and try again. Models are cached in `local_cache/` after first download.

## Under the Hood

### Why fast?

1. Quantized model weights
2. ONNX Runtime for optimized inference on CPU/GPU
3. Parallel batch processing with goroutines

### Why light?

1. No hidden dependencies via Huggingface Transformers
2. Uses efficient Go tokenizer implementation

### Why accurate?

1. Better than OpenAI Ada-002
2. Top of the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features:

- 21 more text embedding models
- Sparse embeddings (SPLADE)
- Image embeddings (CLIP)
- Reranking (cross-encoder)

## License

MIT © [2025](https://github.com/Anush008/fastembed-go/blob/main/LICENSE)

## Credits

- Based on [fastembed-rs](https://github.com/Anush008/fastembed-rs) (Rust implementation)
- Uses [onnxruntime_go](https://github.com/yalue/onnxruntime_go) for ONNX Runtime bindings
- Uses [tokenizer](https://github.com/sugarme/tokenizer) for HuggingFace tokenization
