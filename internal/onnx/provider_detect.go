//go:build !darwin

package onnx

import "github.com/anush008/fastembed-go/cuda"

// detectBestProvider selects the optimal execution provider for non-macOS platforms.
func detectBestProvider(deviceID int) ExecutionProvider {
	// Check for NVIDIA GPU via NVML
	if cuda.IsAvailable() {
		return CUDAProvider(deviceID)
	}

	// Fallback to CPU
	return CPUProvider()
}
