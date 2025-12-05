// Package cuda provides NVIDIA GPU detection and memory management utilities.
package cuda

// MemoryInfo contains GPU memory information in bytes.
type MemoryInfo struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// DeviceInfo contains GPU device information.
type DeviceInfo struct {
	Index  int
	Name   string
	Memory MemoryInfo
}

// TuningParams contains auto-tuned parameters for embedding.
type TuningParams struct {
	BatchSize  int
	MaxWorkers int
}

// ModelMemoryProfile contains memory usage estimates for a model.
type ModelMemoryProfile struct {
	// BaseMemoryMB is the static memory used by the model (weights, etc.)
	BaseMemoryMB int
	// PerBatchItemMB is the estimated memory per batch item
	PerBatchItemMB float64
	// MaxTokens is the maximum sequence length
	MaxTokens int
}

// Known model profiles (estimated values - conservative for safety)
// Memory usage scales with sequence length, so these assume worst-case (max tokens)
var modelProfiles = map[string]ModelMemoryProfile{
	// Small models (384 dim)
	"bge-small-en-v1.5":        {BaseMemoryMB: 200, PerBatchItemMB: 8.0, MaxTokens: 512},
	"all-MiniLM-L6-v2":         {BaseMemoryMB: 150, PerBatchItemMB: 6.0, MaxTokens: 256},
	"paraphrase-MiniLM-L12-v2": {BaseMemoryMB: 200, PerBatchItemMB: 8.0, MaxTokens: 512},
	"multilingual-e5-small":    {BaseMemoryMB: 200, PerBatchItemMB: 8.0, MaxTokens: 512},

	// Medium models (768 dim)
	"bge-base-en-v1.5":             {BaseMemoryMB: 500, PerBatchItemMB: 20.0, MaxTokens: 512},
	"all-mpnet-base-v2":            {BaseMemoryMB: 500, PerBatchItemMB: 18.0, MaxTokens: 384},
	"multilingual-e5-base":         {BaseMemoryMB: 600, PerBatchItemMB: 22.0, MaxTokens: 512},
	"nomic-embed-text-v1":          {BaseMemoryMB: 600, PerBatchItemMB: 50.0, MaxTokens: 8192},
	"nomic-embed-text-v1.5":        {BaseMemoryMB: 600, PerBatchItemMB: 50.0, MaxTokens: 8192},
	"jina-embeddings-v2-base-code": {BaseMemoryMB: 600, PerBatchItemMB: 50.0, MaxTokens: 8192},

	// Large models (768-1024 dim)
	"bge-large-en-v1.5":     {BaseMemoryMB: 1500, PerBatchItemMB: 40.0, MaxTokens: 512},
	"multilingual-e5-large": {BaseMemoryMB: 1500, PerBatchItemMB: 40.0, MaxTokens: 512},
	"mxbai-embed-large-v1":  {BaseMemoryMB: 1500, PerBatchItemMB: 40.0, MaxTokens: 512},
	"gte-large-en-v1.5":     {BaseMemoryMB: 1500, PerBatchItemMB: 80.0, MaxTokens: 8192},

	// Gemma models (768 dim, 300M params) - very memory hungry
	"embedding-gemma-300m": {BaseMemoryMB: 2000, PerBatchItemMB: 150.0, MaxTokens: 2048},
}

// DefaultProfile is used for unknown models (conservative estimates).
var DefaultProfile = ModelMemoryProfile{
	BaseMemoryMB:   800,
	PerBatchItemMB: 50.0,
	MaxTokens:      512,
}

// GetModelProfile returns the memory profile for a model.
func GetModelProfile(modelName string) ModelMemoryProfile {
	if profile, ok := modelProfiles[modelName]; ok {
		return profile
	}
	return DefaultProfile
}

// GetMaxTokens returns the maximum token length for a model.
// This is useful for chunking text to fit within model limits.
func GetMaxTokens(modelName string) int {
	return GetModelProfile(modelName).MaxTokens
}

// Interface definitions for platform-specific implementations.
// These are implemented in cuda_nvml.go (amd64) and cuda_stub.go (other platforms).

// Init initializes the CUDA/NVML library. Safe to call multiple times.
// Implemented per-platform.
// func Init() error

// Shutdown cleans up CUDA/NVML resources.
// Implemented per-platform.
// func Shutdown() error

// IsAvailable returns true if CUDA is available.
// Implemented per-platform.
// func IsAvailable() bool

// DeviceCount returns the number of CUDA devices.
// Implemented per-platform.
// func DeviceCount() (int, error)

// GetDeviceInfo returns information about a specific GPU device.
// Implemented per-platform.
// func GetDeviceInfo(deviceID int) (*DeviceInfo, error)

// GetFreeMemory returns free VRAM in bytes for a specific device.
// Implemented per-platform.
// func GetFreeMemory(deviceID int) (uint64, error)

// AutoTune calculates optimal batch size and workers based on available VRAM.
// Implemented per-platform.
// func AutoTune(deviceID int, modelName string, logger *slog.Logger) (*TuningParams, error)
