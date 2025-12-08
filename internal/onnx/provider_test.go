package onnx_test

import (
	"runtime"
	"testing"

	"github.com/anush008/fastembed-go/cuda"
	"github.com/anush008/fastembed-go/internal/onnx"
)

func TestAutoProviderDetection(t *testing.T) {
	provider := onnx.AutoProviderDefault()
	name := provider.Name()

	t.Logf("Platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	t.Logf("CUDA available: %v", cuda.IsAvailable())
	t.Logf("Selected provider: %s", name)

	// The provider starts as "Auto" before Apply is called
	if name != "Auto" {
		t.Errorf("Expected 'Auto' before Apply, got %s", name)
	}
}

func TestAutoProviderName(t *testing.T) {
	// Test that provider names are correct
	tests := []struct {
		name     string
		provider onnx.ExecutionProvider
		expected string
	}{
		{"CPU", onnx.CPUProvider(), "CPU"},
		{"CoreML-Default", onnx.CoreMLProvider(), "CoreML:CPUAndNeuralEngine"},
		{"CoreML-Safe", onnx.CoreMLProviderWithOptions(onnx.SafeCoreMLOptions()), "CoreML:CPUOnly"},
		{"CoreML-Performance", onnx.CoreMLProviderWithOptions(onnx.PerformanceCoreMLOptions()), "CoreML:ALL"},
		{"CUDA:0", onnx.CUDAProvider(0), "CUDA:0"},
		{"CUDA:1", onnx.CUDAProvider(1), "CUDA:1"},
		{"Auto", onnx.AutoProviderDefault(), "Auto"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.provider.Name(); got != tt.expected {
				t.Errorf("Name() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestCoreMLOptionsDefaults(t *testing.T) {
	opts := onnx.DefaultCoreMLOptions()

	// Check that defaults prioritize precision and stability
	if opts.ModelFormat != onnx.CoreMLModelFormatMLProgram {
		t.Errorf("Expected ModelFormat=MLProgram, got %s", opts.ModelFormat)
	}
	if opts.MLComputeUnits != onnx.CoreMLComputeUnitsCPUAndNeuralEngine {
		t.Errorf("Expected MLComputeUnits=CPUAndNeuralEngine, got %s", opts.MLComputeUnits)
	}
	if opts.AllowLowPrecisionAccumulationOnGPU {
		t.Error("Expected AllowLowPrecisionAccumulationOnGPU=false")
	}
	if opts.RequireStaticInputShapes {
		t.Error("Expected RequireStaticInputShapes=false")
	}
}

func TestCoreMLSafeOptions(t *testing.T) {
	opts := onnx.SafeCoreMLOptions()

	// Safe mode should use CPU only for debugging
	if opts.MLComputeUnits != onnx.CoreMLComputeUnitsCPUOnly {
		t.Errorf("Expected MLComputeUnits=CPUOnly for safe mode, got %s", opts.MLComputeUnits)
	}
	// Other options should remain at safe defaults
	if opts.ModelFormat != onnx.CoreMLModelFormatMLProgram {
		t.Errorf("Expected ModelFormat=MLProgram, got %s", opts.ModelFormat)
	}
	if opts.AllowLowPrecisionAccumulationOnGPU {
		t.Error("Expected AllowLowPrecisionAccumulationOnGPU=false")
	}
}

func TestCoreMLPerformanceOptions(t *testing.T) {
	opts := onnx.PerformanceCoreMLOptions()

	// Performance mode should use all compute units
	if opts.MLComputeUnits != onnx.CoreMLComputeUnitsAll {
		t.Errorf("Expected MLComputeUnits=ALL for performance mode, got %s", opts.MLComputeUnits)
	}
	// MLProgram format should still be used for compatibility
	if opts.ModelFormat != onnx.CoreMLModelFormatMLProgram {
		t.Errorf("Expected ModelFormat=MLProgram, got %s", opts.ModelFormat)
	}
}

func TestCoreMLCustomOptions(t *testing.T) {
	// Test that custom options are properly applied
	customOpts := onnx.CoreMLOptions{
		ModelFormat:                        onnx.CoreMLModelFormatNeuralNetwork,
		MLComputeUnits:                     onnx.CoreMLComputeUnitsCPUAndGPU,
		AllowLowPrecisionAccumulationOnGPU: true,
		RequireStaticInputShapes:           true,
	}

	provider := onnx.CoreMLProviderWithOptions(customOpts)

	// Provider name should reflect the compute units
	expectedName := "CoreML:CPUAndGPU"
	if got := provider.Name(); got != expectedName {
		t.Errorf("Name() = %v, want %v", got, expectedName)
	}
}
