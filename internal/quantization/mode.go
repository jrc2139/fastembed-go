// Package quantization provides quantization mode handling for models.
package quantization

import "fmt"

// Mode represents the quantization mode of a model.
type Mode int

const (
	// None means the model is not quantized (FP32).
	None Mode = iota
	// Static quantization - batch size independent.
	Static
	// Dynamic quantization - batch size must be >= document count.
	Dynamic
)

// String returns the string representation of the quantization mode.
func (m Mode) String() string {
	switch m {
	case None:
		return "none"
	case Static:
		return "static"
	case Dynamic:
		return "dynamic"
	default:
		return "unknown"
	}
}

// ValidateBatchSize checks if the batch size is valid for this quantization mode.
// For dynamic quantization, batch size must be >= document count.
func (m Mode) ValidateBatchSize(batchSize, documentCount int) error {
	if m == Dynamic && batchSize > 0 && batchSize < documentCount {
		return fmt.Errorf(
			"dynamic quantization requires batch size >= document count; "+
				"got batch_size=%d, documents=%d; use batch_size=0 for auto "+
				"or choose a non-quantized model",
			batchSize, documentCount,
		)
	}
	return nil
}

// AdjustBatchSize returns the appropriate batch size for this quantization mode.
// For dynamic quantization, returns document count.
// For other modes, returns the requested batch size or default if 0.
func (m Mode) AdjustBatchSize(requestedBatchSize, documentCount, defaultBatchSize int) int {
	if m == Dynamic {
		return documentCount
	}
	if requestedBatchSize <= 0 {
		return defaultBatchSize
	}
	return requestedBatchSize
}
