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
		{"CoreML", onnx.CoreMLProvider(), "CoreML"},
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
