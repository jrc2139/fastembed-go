//go:build darwin

package onnx

// detectBestProvider selects CoreML on macOS (Apple Silicon).
// CoreML leverages the Neural Engine and Metal GPU for acceleration.
func detectBestProvider(_ int) ExecutionProvider {
	return CoreMLProvider()
}
