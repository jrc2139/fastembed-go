package text

import (
	"github.com/anush008/fastembed-go/internal/output"
	"github.com/anush008/fastembed-go/internal/pooling"
	"github.com/anush008/fastembed-go/internal/quantization"
	"github.com/anush008/fastembed-go/internal/tokenizer"
)

// UserDefinedModel allows users to bring their own ONNX embedding model.
type UserDefinedModel struct {
	// ONNXData is the raw ONNX model bytes.
	ONNXData []byte

	// TokenizerFiles contains the tokenizer configuration files.
	TokenizerFiles tokenizer.Files

	// Dim is the embedding dimension.
	Dim int

	// Pooling overrides the pooling strategy (nil = Mean).
	Pooling *pooling.Strategy

	// Quantization specifies the quantization mode.
	Quantization quantization.Mode

	// OutputKey specifies a custom output key (nil = default precedence).
	OutputKey *output.Key

	// NoTokenTypeIDs indicates the model doesn't use token_type_ids input.
	NoTokenTypeIDs bool
}

// NewUserDefinedModel creates a UserDefinedModel with required fields.
func NewUserDefinedModel(onnxData []byte, tokenizerFiles tokenizer.Files, dim int) *UserDefinedModel {
	return &UserDefinedModel{
		ONNXData:       onnxData,
		TokenizerFiles: tokenizerFiles,
		Dim:            dim,
		Quantization:   quantization.None,
	}
}

// WithPooling sets the pooling strategy.
func (u *UserDefinedModel) WithPooling(p pooling.Strategy) *UserDefinedModel {
	u.Pooling = &p
	return u
}

// WithQuantization sets the quantization mode.
func (u *UserDefinedModel) WithQuantization(q quantization.Mode) *UserDefinedModel {
	u.Quantization = q
	return u
}

// WithOutputKey sets a custom output key.
func (u *UserDefinedModel) WithOutputKey(k output.Key) *UserDefinedModel {
	u.OutputKey = &k
	return u
}

// WithNoTokenTypeIDs indicates the model doesn't use token_type_ids.
func (u *UserDefinedModel) WithNoTokenTypeIDs() *UserDefinedModel {
	u.NoTokenTypeIDs = true
	return u
}

// getPooling returns the effective pooling strategy.
func (u *UserDefinedModel) getPooling() pooling.Strategy {
	if u.Pooling != nil {
		return *u.Pooling
	}
	return pooling.Mean
}

// getOutputKey returns the output key or default.
func (u *UserDefinedModel) getOutputKey() string {
	if u.OutputKey != nil && u.OutputKey.Type == output.ByName {
		return u.OutputKey.Name
	}
	return "last_hidden_state"
}
