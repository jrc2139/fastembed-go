// Package model provides model information and registry infrastructure.
package model

import "github.com/anush008/fastembed-go/internal/output"

// Info contains metadata for any embedding model.
// It is generic over the model enum type M.
type Info[M comparable] struct {
	// Model is the model identifier (enum value).
	Model M

	// Dim is the embedding dimension.
	Dim int

	// Description is a human-readable description.
	Description string

	// ModelCode is the HuggingFace repository ID.
	ModelCode string

	// ModelFile is the ONNX model file path within the repository.
	ModelFile string

	// AdditionalFiles are extra files to download (e.g., .onnx_data files).
	AdditionalFiles []string

	// TokenizerPath is the subdirectory containing tokenizer files (empty for root).
	TokenizerPath string

	// OutputKey specifies a custom output key (nil for default precedence).
	OutputKey *output.Key

	// NoTokenTypeIDs indicates the model doesn't use token_type_ids input.
	NoTokenTypeIDs bool
}

// RerankerInfo contains metadata for reranking models (no embedding dimension).
type RerankerInfo[M comparable] struct {
	// Model is the model identifier.
	Model M

	// Description is a human-readable description.
	Description string

	// ModelCode is the HuggingFace repository ID.
	ModelCode string

	// ModelFile is the ONNX model file path.
	ModelFile string

	// AdditionalFiles are extra files to download.
	AdditionalFiles []string

	// TokenizerPath is the subdirectory containing tokenizer files.
	TokenizerPath string

	// NoTokenTypeIDs indicates the model doesn't use token_type_ids input.
	NoTokenTypeIDs bool
}

// ImageInfo contains metadata for image embedding models.
type ImageInfo[M comparable] struct {
	// Model is the model identifier.
	Model M

	// Dim is the embedding dimension.
	Dim int

	// Description is a human-readable description.
	Description string

	// ModelCode is the HuggingFace repository ID.
	ModelCode string

	// ModelFile is the ONNX model file path.
	ModelFile string

	// AdditionalFiles are extra files to download.
	AdditionalFiles []string

	// OutputKey specifies a custom output key.
	OutputKey *output.Key
}
