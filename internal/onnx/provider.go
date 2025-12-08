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

// CoreMLComputeUnits specifies which hardware units to use for CoreML execution.
type CoreMLComputeUnits string

const (
	// CoreMLComputeUnitsAll enables all available compute units (CPU, GPU, Neural Engine).
	CoreMLComputeUnitsAll CoreMLComputeUnits = "ALL"
	// CoreMLComputeUnitsCPUOnly restricts execution to CPU only.
	CoreMLComputeUnitsCPUOnly CoreMLComputeUnits = "CPUOnly"
	// CoreMLComputeUnitsCPUAndGPU enables CPU and GPU acceleration.
	CoreMLComputeUnitsCPUAndGPU CoreMLComputeUnits = "CPUAndGPU"
	// CoreMLComputeUnitsCPUAndNeuralEngine enables CPU and Neural Engine (recommended).
	CoreMLComputeUnitsCPUAndNeuralEngine CoreMLComputeUnits = "CPUAndNeuralEngine"
)

// CoreMLModelFormat specifies the CoreML model format.
type CoreMLModelFormat string

const (
	// CoreMLModelFormatNeuralNetwork uses the legacy NeuralNetwork format (Core ML 3+).
	CoreMLModelFormatNeuralNetwork CoreMLModelFormat = "NeuralNetwork"
	// CoreMLModelFormatMLProgram uses the modern MLProgram format (Core ML 5+, macOS 12+).
	CoreMLModelFormatMLProgram CoreMLModelFormat = "MLProgram"
)

// CoreMLOptions configures the CoreML execution provider.
type CoreMLOptions struct {
	// ModelFormat specifies the CoreML model format.
	// MLProgram requires Core ML 5+ (macOS 12+, iOS 15+) but has better precision.
	// Default: MLProgram
	ModelFormat CoreMLModelFormat

	// MLComputeUnits specifies which hardware to use for acceleration.
	// CPUAndNeuralEngine is recommended to avoid GPU precision issues.
	// Default: CPUAndNeuralEngine
	MLComputeUnits CoreMLComputeUnits

	// AllowLowPrecisionAccumulationOnGPU controls whether to use float16
	// accumulation on GPU. Set to false to prevent NaN issues.
	// Default: false
	AllowLowPrecisionAccumulationOnGPU bool

	// RequireStaticInputShapes controls whether to require static input shapes.
	// Dynamic shapes may impact performance but allow variable batch sizes.
	// Default: false
	RequireStaticInputShapes bool
}

// DefaultCoreMLOptions returns safe default options for CoreML.
// These defaults prioritize precision and stability:
// - MLProgram format for better precision handling
// - CPUAndNeuralEngine to avoid GPU precision issues
// - Low precision accumulation disabled
func DefaultCoreMLOptions() CoreMLOptions {
	return CoreMLOptions{
		ModelFormat:                        CoreMLModelFormatMLProgram,
		MLComputeUnits:                     CoreMLComputeUnitsCPUAndNeuralEngine,
		AllowLowPrecisionAccumulationOnGPU: false,
		RequireStaticInputShapes:           false,
	}
}

// SafeCoreMLOptions returns the most conservative options for debugging.
// Uses CPU only to isolate any precision issues from Neural Engine or GPU.
func SafeCoreMLOptions() CoreMLOptions {
	opts := DefaultCoreMLOptions()
	opts.MLComputeUnits = CoreMLComputeUnitsCPUOnly
	return opts
}

// PerformanceCoreMLOptions returns options optimized for maximum performance.
// Uses all available compute units which may have slight precision tradeoffs.
func PerformanceCoreMLOptions() CoreMLOptions {
	opts := DefaultCoreMLOptions()
	opts.MLComputeUnits = CoreMLComputeUnitsAll
	return opts
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

// CoreMLProvider returns a CoreML execution provider with precision-safe defaults.
// Uses MLProgram format and CPUAndNeuralEngine to avoid NaN issues.
func CoreMLProvider() ExecutionProvider {
	return &coreMLProvider{options: DefaultCoreMLOptions()}
}

// CoreMLProviderWithOptions returns a CoreML execution provider with custom options.
func CoreMLProviderWithOptions(opts CoreMLOptions) ExecutionProvider {
	return &coreMLProvider{options: opts}
}

type coreMLProvider struct {
	options CoreMLOptions
}

func (p *coreMLProvider) Apply(opts *ort.SessionOptions) error {
	// Build options map for V2 API
	coreMLOpts := map[string]string{
		"ModelFormat":                        string(p.options.ModelFormat),
		"MLComputeUnits":                     string(p.options.MLComputeUnits),
		"AllowLowPrecisionAccumulationOnGPU": boolToString(p.options.AllowLowPrecisionAccumulationOnGPU),
		"RequireStaticInputShapes":           boolToString(p.options.RequireStaticInputShapes),
	}

	// Use V2 API for proper configuration
	if err := opts.AppendExecutionProviderCoreMLV2(coreMLOpts); err != nil {
		return fmt.Errorf("failed to append CoreML provider: %w", err)
	}
	return nil
}

func (p *coreMLProvider) Name() string {
	return fmt.Sprintf("CoreML:%s", p.options.MLComputeUnits)
}

// boolToString converts a boolean to "0" or "1" for ONNX Runtime options.
func boolToString(b bool) string {
	if b {
		return "1"
	}
	return "0"
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
