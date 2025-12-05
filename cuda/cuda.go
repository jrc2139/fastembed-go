// Package cuda provides NVIDIA GPU detection and memory management utilities.
package cuda

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

var (
	initOnce sync.Once
	initErr  error
	nvmlInit bool
)

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

// Init initializes the NVML library. Safe to call multiple times.
func Init() error {
	initOnce.Do(func() {
		ret := nvml.Init()
		if ret != nvml.SUCCESS {
			initErr = fmt.Errorf("failed to initialize NVML: %s", nvml.ErrorString(ret))
			return
		}
		nvmlInit = true
	})
	return initErr
}

// Shutdown cleans up NVML resources.
func Shutdown() error {
	if !nvmlInit {
		return nil
	}
	ret := nvml.Shutdown()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("failed to shutdown NVML: %s", nvml.ErrorString(ret))
	}
	return nil
}

// IsAvailable returns true if CUDA is available.
func IsAvailable() bool {
	if err := Init(); err != nil {
		return false
	}
	count, ret := nvml.DeviceGetCount()
	return ret == nvml.SUCCESS && count > 0
}

// DeviceCount returns the number of CUDA devices.
func DeviceCount() (int, error) {
	if err := Init(); err != nil {
		return 0, err
	}
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return 0, fmt.Errorf("failed to get device count: %s", nvml.ErrorString(ret))
	}
	return count, nil
}

// GetDeviceInfo returns information about a specific GPU device.
func GetDeviceInfo(deviceID int) (*DeviceInfo, error) {
	if err := Init(); err != nil {
		return nil, err
	}

	device, ret := nvml.DeviceGetHandleByIndex(deviceID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("failed to get device %d: %s", deviceID, nvml.ErrorString(ret))
	}

	name, ret := device.GetName()
	if ret != nvml.SUCCESS {
		name = "Unknown"
	}

	memory, ret := device.GetMemoryInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("failed to get memory info for device %d: %s", deviceID, nvml.ErrorString(ret))
	}

	return &DeviceInfo{
		Index: deviceID,
		Name:  name,
		Memory: MemoryInfo{
			Total: memory.Total,
			Free:  memory.Free,
			Used:  memory.Used,
		},
	}, nil
}

// GetFreeMemory returns free VRAM in bytes for a specific device.
func GetFreeMemory(deviceID int) (uint64, error) {
	info, err := GetDeviceInfo(deviceID)
	if err != nil {
		return 0, err
	}
	return info.Memory.Free, nil
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

// AutoTune calculates optimal batch size and workers based on available VRAM.
func AutoTune(deviceID int, modelName string, logger *slog.Logger) (*TuningParams, error) {
	freeMemory, err := GetFreeMemory(deviceID)
	if err != nil {
		return nil, err
	}

	profile := GetModelProfile(modelName)
	freeMemoryMB := int(freeMemory / 1024 / 1024)

	// Reserve memory for model weights and overhead (use 70% of free memory)
	usableMemoryMB := int(float64(freeMemoryMB) * 0.7)
	availableForBatchesMB := usableMemoryMB - profile.BaseMemoryMB

	if availableForBatchesMB < int(profile.PerBatchItemMB) {
		return nil, fmt.Errorf("insufficient VRAM: need at least %dMB, have %dMB free",
			profile.BaseMemoryMB+int(profile.PerBatchItemMB), freeMemoryMB)
	}

	// Calculate optimal batch size
	batchSize := int(float64(availableForBatchesMB) / profile.PerBatchItemMB)

	// Clamp batch size to reasonable range
	if batchSize < 1 {
		batchSize = 1
	} else if batchSize > 64 {
		batchSize = 64 // Cap at 64 to avoid memory fragmentation issues
	}

	// Calculate max workers - for GPU, fewer is usually better to avoid contention
	// Use 1-2 workers for very constrained memory, up to 4 for plenty of memory
	maxWorkers := 1
	if availableForBatchesMB > profile.BaseMemoryMB*2 {
		maxWorkers = 2
	}
	if availableForBatchesMB > profile.BaseMemoryMB*4 {
		maxWorkers = 4
	}

	if logger != nil {
		logger.Info("auto-tuned embedding parameters",
			"device", deviceID,
			"model", modelName,
			"free_vram_mb", freeMemoryMB,
			"usable_mb", usableMemoryMB,
			"base_model_mb", profile.BaseMemoryMB,
			"per_batch_mb", profile.PerBatchItemMB,
			"batch_size", batchSize,
			"max_workers", maxWorkers,
		)
	}

	return &TuningParams{
		BatchSize:  batchSize,
		MaxWorkers: maxWorkers,
	}, nil
}
