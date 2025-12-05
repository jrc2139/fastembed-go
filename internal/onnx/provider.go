// Package onnx provides ONNX Runtime session management.
package onnx

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// ExecutionProvider configures an ONNX execution provider.
type ExecutionProvider interface {
	// Apply configures the execution provider on session options.
	Apply(opts *ort.SessionOptions) error

	// Name returns the provider name.
	Name() string
}

// CPUProvider returns a CPU execution provider.
func CPUProvider() ExecutionProvider {
	return &cpuProvider{}
}

type cpuProvider struct{}

func (p *cpuProvider) Apply(opts *ort.SessionOptions) error {
	// CPU is the default, no configuration needed
	return nil
}

func (p *cpuProvider) Name() string {
	return "CPU"
}

// CUDAProvider returns a CUDA execution provider for GPU acceleration.
func CUDAProvider(deviceID int) ExecutionProvider {
	return &cudaProvider{deviceID: deviceID}
}

type cudaProvider struct {
	deviceID int
}

func (p *cudaProvider) Apply(opts *ort.SessionOptions) error {
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return fmt.Errorf("failed to create CUDA options: %w", err)
	}
	defer cudaOpts.Destroy()

	if err := cudaOpts.Update(map[string]string{
		"device_id": fmt.Sprintf("%d", p.deviceID),
	}); err != nil {
		return fmt.Errorf("failed to set CUDA device: %w", err)
	}

	if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		return fmt.Errorf("failed to append CUDA provider: %w", err)
	}

	return nil
}

func (p *cudaProvider) Name() string {
	return fmt.Sprintf("CUDA:%d", p.deviceID)
}

// CoreMLProvider returns a CoreML execution provider for Apple Silicon.
func CoreMLProvider() ExecutionProvider {
	return &coreMLProvider{}
}

type coreMLProvider struct{}

func (p *coreMLProvider) Apply(opts *ort.SessionOptions) error {
	if err := opts.AppendExecutionProviderCoreML(0); err != nil {
		return fmt.Errorf("failed to append CoreML provider: %w", err)
	}
	return nil
}

func (p *coreMLProvider) Name() string {
	return "CoreML"
}

// autoProvider automatically selects the best available execution provider.
type autoProvider struct {
	deviceID int
	selected ExecutionProvider
}

// AutoProvider returns an execution provider that automatically selects
// the best available backend for the current platform:
//   - macOS: CoreML (leverages Neural Engine + Metal GPU)
//   - Linux/Windows with NVIDIA GPU: CUDA
//   - Fallback: CPU
//
// The deviceID parameter is used for CUDA device selection (ignored on other platforms).
func AutoProvider(deviceID int) ExecutionProvider {
	return &autoProvider{deviceID: deviceID}
}

// AutoProviderDefault returns AutoProvider with device 0.
func AutoProviderDefault() ExecutionProvider {
	return AutoProvider(0)
}

func (p *autoProvider) Apply(opts *ort.SessionOptions) error {
	p.selected = detectBestProvider(p.deviceID)
	return p.selected.Apply(opts)
}

func (p *autoProvider) Name() string {
	if p.selected != nil {
		return "Auto:" + p.selected.Name()
	}
	return "Auto"
}
