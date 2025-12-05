//go:build !((linux || windows) && amd64)

// Package cuda provides NVIDIA GPU detection and memory management utilities.
// This file contains stub implementations for platforms without NVML support.
package cuda

import (
	"errors"
	"log/slog"
)

// ErrNotSupported is returned when CUDA/NVML is not supported on the platform.
var ErrNotSupported = errors.New("CUDA/NVML not supported on this platform")

// Init is a no-op on non-amd64 platforms.
func Init() error {
	return nil
}

// Shutdown is a no-op on non-amd64 platforms.
func Shutdown() error {
	return nil
}

// IsAvailable returns false on non-amd64 platforms (no CUDA support).
func IsAvailable() bool {
	return false
}

// DeviceCount returns 0 on non-amd64 platforms.
func DeviceCount() (int, error) {
	return 0, nil
}

// GetDeviceInfo returns an error on non-amd64 platforms.
func GetDeviceInfo(deviceID int) (*DeviceInfo, error) {
	return nil, ErrNotSupported
}

// GetFreeMemory returns an error on non-amd64 platforms.
func GetFreeMemory(deviceID int) (uint64, error) {
	return 0, ErrNotSupported
}

// AutoTune returns an error on non-amd64 platforms.
func AutoTune(deviceID int, modelName string, logger *slog.Logger) (*TuningParams, error) {
	return nil, ErrNotSupported
}
