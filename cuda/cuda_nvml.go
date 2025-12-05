//go:build (linux || windows) && amd64

// Package cuda provides NVIDIA GPU detection and memory management utilities.
// This file contains NVML-based implementations for amd64 platforms.
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
