// Package fastembed provides v2 re-exports for convenient access to text embeddings.
//
// The text package can be imported directly for the full API:
//
//	import "github.com/anush008/fastembed-go/text"
//
// Or use these re-exports for convenience:
//
//	import fastembed "github.com/anush008/fastembed-go"
//	emb, _ := fastembed.NewTextEmbedding(fastembed.TextWithModel(fastembed.TextAllMiniLML6V2))
package fastembed

import (
	"context"

	"github.com/anush008/fastembed-go/internal/model"
	"github.com/anush008/fastembed-go/internal/pooling"
	"github.com/anush008/fastembed-go/text"
)

// TextEmbedding is the v2 text embedding type.
// Use NewTextEmbedding to create a new instance.
type TextEmbedding = text.TextEmbedding

// TextEmbedder is an interface for text embedding functionality.
type TextEmbedder interface {
	Embed(ctx context.Context, texts []string, batchSize int) ([][]float32, error)
	QueryEmbed(ctx context.Context, query string) ([]float32, error)
	PassageEmbed(ctx context.Context, passages []string, batchSize int) ([][]float32, error)
	Destroy() error
}

// TextModel represents a text embedding model (v2).
type TextModel = text.Model

// Text model constants (v2).
const (
	// MiniLM models
	TextAllMiniLML6V2  = text.AllMiniLML6V2
	TextAllMiniLML6V2Q = text.AllMiniLML6V2Q
	TextAllMiniLML12V2 = text.AllMiniLML12V2

	// BGE models
	TextBGESmallEN     = text.BGESmallEN
	TextBGESmallENV15  = text.BGESmallENV15
	TextBGESmallENV15Q = text.BGESmallENV15Q
	TextBGEBaseEN      = text.BGEBaseEN
	TextBGEBaseENV15   = text.BGEBaseENV15
	TextBGEBaseENV15Q  = text.BGEBaseENV15Q
	TextBGELargeENV15  = text.BGELargeENV15
	TextBGELargeENV15Q = text.BGELargeENV15Q
	TextBGESmallZH     = text.BGESmallZH

	// Multilingual E5 models
	TextMultilingualE5Small         = text.MultilingualE5Small
	TextMultilingualE5Base          = text.MultilingualE5Base
	TextMultilingualE5Large         = text.MultilingualE5Large
	TextMultilingualE5LargeInstruct = text.MultilingualE5LargeInstruct

	// Nomic models
	TextNomicEmbedTextV1   = text.NomicEmbedTextV1
	TextNomicEmbedTextV15  = text.NomicEmbedTextV15
	TextNomicEmbedTextV15Q = text.NomicEmbedTextV15Q

	// MixedBread models
	TextMxbaiEmbedLargeV1  = text.MxbaiEmbedLargeV1
	TextMxbaiEmbedLargeV1Q = text.MxbaiEmbedLargeV1Q

	// GTE models
	TextGTEBaseENV15  = text.GTEBaseENV15
	TextGTEBaseENV15Q = text.GTEBaseENV15Q
	TextGTELargeENV15 = text.GTELargeENV15

	// EmbeddingGemma models
	TextEmbeddingGemma300M   = text.EmbeddingGemma300M
	TextEmbeddingGemma300MQ4 = text.EmbeddingGemma300MQ4

	// Other models
	TextParaphraseMLMiniLML12V2  = text.ParaphraseMLMiniLML12V2
	TextParaphraseMLMiniLML12V2Q = text.ParaphraseMLMiniLML12V2Q
	TextJinaEmbeddingsV2BaseCode = text.JinaEmbeddingsV2BaseCode
)

// TextOption configures TextEmbedding (v2).
type TextOption = text.Option

// TextWithModel sets the model for TextEmbedding.
func TextWithModel(m TextModel) TextOption { return text.WithModel(m) }

// TextWithCacheDir sets the cache directory.
func TextWithCacheDir(dir string) TextOption { return text.WithCacheDir(dir) }

// TextWithMaxLength sets the maximum sequence length.
func TextWithMaxLength(n int) TextOption { return text.WithMaxLength(n) }

// TextWithCUDA enables CUDA on the specified device.
func TextWithCUDA(deviceID int) TextOption { return text.WithCUDA(deviceID) }

// TextWithPooling overrides the model's default pooling.
func TextWithPooling(p PoolingV2) TextOption { return text.WithPooling(p) }

// TextWithShowDownloadProgress controls progress display.
func TextWithShowDownloadProgress(show bool) TextOption { return text.WithShowDownloadProgress(show) }

// NewTextEmbedding creates a v2 TextEmbedding with the given options.
func NewTextEmbedding(opts ...TextOption) (*TextEmbedding, error) {
	return text.New(opts...)
}

// ListTextModels returns all available text embedding models (v2).
func ListTextModels() []*model.Info[TextModel] {
	return text.ListModels()
}

// GetTextModelInfo returns info for a specific model.
func GetTextModelInfo(m TextModel) *model.Info[TextModel] {
	return text.GetModelInfo(m)
}

// PoolingV2 is the pooling strategy type (v2).
type PoolingV2 = pooling.Strategy

// Pooling strategy constants (v2).
const (
	PoolingV2Cls  = pooling.Cls
	PoolingV2Mean = pooling.Mean
)
