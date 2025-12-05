package text

import (
	"log/slog"
	"os"
	"runtime"

	"github.com/anush008/fastembed-go/internal/download"
	"github.com/anush008/fastembed-go/internal/onnx"
	"github.com/anush008/fastembed-go/internal/pooling"
)

// config holds all configuration for TextEmbedding.
type config struct {
	Model        Model
	CacheDir     string
	MaxLength    int
	ShowProgress bool
	Pooling      *pooling.Strategy
	Providers    []onnx.ExecutionProvider
	Logger       *slog.Logger
	MaxWorkers   int  // Maximum number of worker goroutines for parallel batch processing
	AutoTune     bool // Automatically tune batch size and workers based on available VRAM
	CUDADeviceID int  // CUDA device ID for auto-tuning (default 0)
}

// defaultLogger returns a default JSON logger to stderr at INFO level.
func defaultLogger() *slog.Logger {
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
}

// defaultConfig returns the default configuration.
func defaultConfig() config {
	return config{
		Model:        BGESmallENV15,
		CacheDir:     download.GetCacheDir(),
		MaxLength:    512,
		ShowProgress: true,
		Logger:       defaultLogger(),
		MaxWorkers:   runtime.NumCPU(),
		// Auto-detect best execution provider:
		// - macOS: CoreML (Neural Engine + Metal)
		// - Linux/Windows + NVIDIA: CUDA
		// - Fallback: CPU
		Providers: []onnx.ExecutionProvider{onnx.AutoProviderDefault()},
	}
}

// Option configures TextEmbedding.
type Option func(*config)

// WithModel sets the embedding model.
func WithModel(m Model) Option {
	return func(c *config) {
		c.Model = m
	}
}

// WithCacheDir sets the model cache directory.
func WithCacheDir(dir string) Option {
	return func(c *config) {
		c.CacheDir = dir
	}
}

// WithMaxLength sets the maximum sequence length for tokenization.
func WithMaxLength(n int) Option {
	return func(c *config) {
		c.MaxLength = n
	}
}

// WithShowDownloadProgress enables or disables download progress display.
func WithShowDownloadProgress(show bool) Option {
	return func(c *config) {
		c.ShowProgress = show
	}
}

// WithPooling overrides the model's default pooling strategy.
func WithPooling(p pooling.Strategy) Option {
	return func(c *config) {
		c.Pooling = &p
	}
}

// WithExecutionProviders sets the ONNX execution providers.
func WithExecutionProviders(providers ...onnx.ExecutionProvider) Option {
	return func(c *config) {
		c.Providers = providers
	}
}

// WithCUDA enables CUDA execution on the specified GPU device.
func WithCUDA(deviceID int) Option {
	return func(c *config) {
		c.Providers = append(c.Providers, onnx.CUDAProvider(deviceID))
	}
}

// WithCoreML enables CoreML execution (Apple Silicon).
func WithCoreML() Option {
	return func(c *config) {
		c.Providers = append(c.Providers, onnx.CoreMLProvider())
	}
}

// WithAuto enables automatic execution provider selection based on platform:
//   - macOS: CoreML (leverages Neural Engine + Metal GPU)
//   - Linux/Windows with NVIDIA GPU: CUDA
//   - Fallback: CPU
//
// Note: This is the default behavior. Use WithCPU() to explicitly disable
// hardware acceleration and use CPU only.
func WithAuto() Option {
	return func(c *config) {
		c.Providers = []onnx.ExecutionProvider{onnx.AutoProviderDefault()}
	}
}

// WithCPU explicitly disables hardware acceleration and uses CPU only.
// Use this to override the default auto-detection behavior.
func WithCPU() Option {
	return func(c *config) {
		c.Providers = []onnx.ExecutionProvider{onnx.CPUProvider()}
	}
}

// WithLogger sets a custom slog.Logger for the embedding model.
// If not set, a default JSON logger to stderr is used.
func WithLogger(logger *slog.Logger) Option {
	return func(c *config) {
		c.Logger = logger
	}
}

// WithMaxWorkers sets the maximum number of worker goroutines for parallel batch processing.
// Default is runtime.NumCPU().
func WithMaxWorkers(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.MaxWorkers = n
		}
	}
}

// WithAutoTune enables automatic tuning of batch size and max workers
// based on available GPU VRAM. This is recommended when using CUDA.
// When enabled, it queries the GPU memory and calculates optimal parameters
// for the selected model to avoid out-of-memory errors.
func WithAutoTune(deviceID int) Option {
	return func(c *config) {
		c.AutoTune = true
		c.CUDADeviceID = deviceID
	}
}
