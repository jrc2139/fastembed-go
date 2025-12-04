// Package text provides dense text embedding generation.
package text

import (
	"github.com/anush008/fastembed-go/internal/model"
	"github.com/anush008/fastembed-go/internal/output"
	"github.com/anush008/fastembed-go/internal/pooling"
	"github.com/anush008/fastembed-go/internal/quantization"
)

// Model represents a supported text embedding model.
type Model string

// Supported text embedding models.
const (
	// MiniLM models - Small, fast, good quality
	AllMiniLML6V2  Model = "all-MiniLM-L6-v2"
	AllMiniLML6V2Q Model = "all-MiniLM-L6-v2-q" // Dynamic quantized
	AllMiniLML12V2 Model = "all-MiniLM-L12-v2"

	// BGE models - BAAI General Embedding
	BGESmallENV15  Model = "bge-small-en-v1.5"
	BGESmallENV15Q Model = "bge-small-en-v1.5-q" // Static quantized
	BGEBaseENV15   Model = "bge-base-en-v1.5"
	BGEBaseENV15Q  Model = "bge-base-en-v1.5-q" // Static quantized
	BGELargeENV15  Model = "bge-large-en-v1.5"
	BGELargeENV15Q Model = "bge-large-en-v1.5-q" // Static quantized
	BGESmallEN     Model = "bge-small-en"
	BGEBaseEN      Model = "bge-base-en"
	BGESmallZH     Model = "bge-small-zh-v1.5"

	// Multilingual E5 models - 100+ languages
	MultilingualE5Small         Model = "multilingual-e5-small"
	MultilingualE5Base          Model = "multilingual-e5-base"
	MultilingualE5Large         Model = "multilingual-e5-large"
	MultilingualE5LargeInstruct Model = "multilingual-e5-large-instruct"

	// Google EmbeddingGemma - Matryoshka support
	EmbeddingGemma300M   Model = "embedding-gemma-300m"
	EmbeddingGemma300MQ4 Model = "embedding-gemma-300m-q4"

	// Nomic models - Matryoshka support
	NomicEmbedTextV1   Model = "nomic-embed-text-v1"
	NomicEmbedTextV15  Model = "nomic-embed-text-v1.5"
	NomicEmbedTextV15Q Model = "nomic-embed-text-v1.5-q" // Dynamic quantized

	// MixedBread models
	MxbaiEmbedLargeV1  Model = "mxbai-embed-large-v1"
	MxbaiEmbedLargeV1Q Model = "mxbai-embed-large-v1-q" // Dynamic quantized

	// GTE models
	GTEBaseENV15  Model = "gte-base-en-v1.5"
	GTEBaseENV15Q Model = "gte-base-en-v1.5-q" // Dynamic quantized
	GTELargeENV15 Model = "gte-large-en-v1.5"

	// Other models
	ParaphraseMLMiniLML12V2  Model = "paraphrase-multilingual-MiniLM-L12-v2"
	ParaphraseMLMiniLML12V2Q Model = "paraphrase-multilingual-MiniLM-L12-v2-q"
	JinaEmbeddingsV2BaseCode Model = "jina-embeddings-v2-base-code"
)

// registry holds all model metadata.
var registry = model.NewRegistry[Model, model.Info[Model]]()

func init() {
	initRegistry()
}

// initRegistry populates the model registry.
func initRegistry() {
	// Helper for output key
	sentenceEmb := output.NewKeyByName("sentence_embedding")

	models := map[Model]*model.Info[Model]{
		// MiniLM models
		AllMiniLML6V2: {
			Model:       AllMiniLML6V2,
			Dim:         384,
			Description: "Sentence Transformer MiniLM-L6-v2",
			ModelCode:   "Qdrant/all-MiniLM-L6-v2-onnx",
			ModelFile:   "model.onnx",
		},
		AllMiniLML6V2Q: {
			Model:       AllMiniLML6V2Q,
			Dim:         384,
			Description: "Sentence Transformer MiniLM-L6-v2 (Quantized)",
			ModelCode:   "Qdrant/all-MiniLM-L6-v2-onnx",
			ModelFile:   "model_optimized.onnx",
		},
		AllMiniLML12V2: {
			Model:       AllMiniLML12V2,
			Dim:         384,
			Description: "Sentence Transformer MiniLM-L12-v2",
			ModelCode:   "Qdrant/all-MiniLM-L12-v2-onnx",
			ModelFile:   "model.onnx",
		},

		// BGE models
		BGESmallENV15: {
			Model:       BGESmallENV15,
			Dim:         384,
			Description: "Fast and Default English model",
			ModelCode:   "Qdrant/bge-small-en-v1.5-onnx-Q",
			ModelFile:   "model_optimized.onnx",
		},
		BGESmallENV15Q: {
			Model:       BGESmallENV15Q,
			Dim:         384,
			Description: "Fast and Default English model (Quantized)",
			ModelCode:   "Qdrant/bge-small-en-v1.5-onnx-Q",
			ModelFile:   "model_optimized.onnx",
		},
		BGEBaseENV15: {
			Model:       BGEBaseENV15,
			Dim:         768,
			Description: "Base English v1.5 model",
			ModelCode:   "Qdrant/bge-base-en-v1.5-onnx-Q",
			ModelFile:   "model_optimized.onnx",
		},
		BGEBaseENV15Q: {
			Model:       BGEBaseENV15Q,
			Dim:         768,
			Description: "Base English v1.5 model (Quantized)",
			ModelCode:   "Qdrant/bge-base-en-v1.5-onnx-Q",
			ModelFile:   "model_optimized.onnx",
		},
		BGELargeENV15: {
			Model:       BGELargeENV15,
			Dim:         1024,
			Description: "Large English v1.5 model",
			ModelCode:   "Qdrant/bge-large-en-v1.5-onnx",
			ModelFile:   "model.onnx",
		},
		BGELargeENV15Q: {
			Model:       BGELargeENV15Q,
			Dim:         1024,
			Description: "Large English v1.5 model (Quantized)",
			ModelCode:   "Qdrant/bge-large-en-v1.5-onnx-Q",
			ModelFile:   "model_optimized.onnx",
		},
		BGESmallEN: {
			Model:       BGESmallEN,
			Dim:         384,
			Description: "Fast English model",
			ModelCode:   "Qdrant/bge-small-en-onnx",
			ModelFile:   "model_optimized.onnx",
		},
		BGEBaseEN: {
			Model:       BGEBaseEN,
			Dim:         768,
			Description: "Base English model",
			ModelCode:   "Qdrant/bge-base-en-onnx",
			ModelFile:   "model_optimized.onnx",
		},
		BGESmallZH: {
			Model:       BGESmallZH,
			Dim:         512,
			Description: "Fast Chinese model",
			ModelCode:   "Qdrant/bge-small-zh-v1.5-onnx",
			ModelFile:   "model.onnx",
		},

		// Multilingual E5 models
		MultilingualE5Small: {
			Model:          MultilingualE5Small,
			Dim:            384,
			Description:    "Multilingual E5 Small - 100+ languages",
			ModelCode:      "Qdrant/multilingual-e5-small-onnx",
			ModelFile:      "model.onnx",
			NoTokenTypeIDs: true,
		},
		MultilingualE5Base: {
			Model:          MultilingualE5Base,
			Dim:            768,
			Description:    "Multilingual E5 Base - 100+ languages",
			ModelCode:      "Qdrant/multilingual-e5-base-onnx",
			ModelFile:      "model.onnx",
			NoTokenTypeIDs: true,
		},
		MultilingualE5Large: {
			Model:           MultilingualE5Large,
			Dim:             1024,
			Description:     "Multilingual E5 Large - 100+ languages",
			ModelCode:       "Qdrant/multilingual-e5-large-onnx",
			ModelFile:       "model.onnx",
			AdditionalFiles: []string{"model.onnx_data"},
			NoTokenTypeIDs:  true,
		},
		MultilingualE5LargeInstruct: {
			Model:           MultilingualE5LargeInstruct,
			Dim:             1024,
			Description:     "Multilingual E5 Large Instruct - task-specific",
			ModelCode:       "intfloat/multilingual-e5-large-instruct",
			ModelFile:       "onnx/model.onnx",
			AdditionalFiles: []string{"onnx/model.onnx_data"},
			TokenizerPath:   "onnx",
			OutputKey:       sentenceEmb,
			NoTokenTypeIDs:  true,
		},

		// EmbeddingGemma models
		EmbeddingGemma300M: {
			Model:           EmbeddingGemma300M,
			Dim:             768,
			Description:     "Google EmbeddingGemma 300M - Matryoshka support",
			ModelCode:       "onnx-community/embeddinggemma-300m-ONNX",
			ModelFile:       "onnx/model.onnx",
			AdditionalFiles: []string{"onnx/model.onnx_data"},
			TokenizerPath:   "onnx",
			OutputKey:       sentenceEmb,
			NoTokenTypeIDs:  true,
		},
		EmbeddingGemma300MQ4: {
			Model:           EmbeddingGemma300MQ4,
			Dim:             768,
			Description:     "Google EmbeddingGemma 300M Q4 - Quantized",
			ModelCode:       "onnx-community/embeddinggemma-300m-ONNX",
			ModelFile:       "onnx/model_q4.onnx",
			AdditionalFiles: []string{"onnx/model_q4.onnx_data"},
			TokenizerPath:   "onnx",
			OutputKey:       sentenceEmb,
			NoTokenTypeIDs:  true,
		},

		// Nomic models
		NomicEmbedTextV1: {
			Model:          NomicEmbedTextV1,
			Dim:            768,
			Description:    "Nomic Embed Text v1",
			ModelCode:      "nomic-ai/nomic-embed-text-v1",
			ModelFile:      "onnx/model.onnx",
			TokenizerPath:  "onnx",
			NoTokenTypeIDs: true,
		},
		NomicEmbedTextV15: {
			Model:          NomicEmbedTextV15,
			Dim:            768,
			Description:    "Nomic Embed Text v1.5 - Matryoshka support",
			ModelCode:      "nomic-ai/nomic-embed-text-v1.5",
			ModelFile:      "onnx/model.onnx",
			TokenizerPath:  "onnx",
			NoTokenTypeIDs: true,
		},
		NomicEmbedTextV15Q: {
			Model:          NomicEmbedTextV15Q,
			Dim:            768,
			Description:    "Nomic Embed Text v1.5 (Quantized)",
			ModelCode:      "nomic-ai/nomic-embed-text-v1.5",
			ModelFile:      "onnx/model_quantized.onnx",
			TokenizerPath:  "onnx",
			NoTokenTypeIDs: true,
		},

		// MixedBread models
		MxbaiEmbedLargeV1: {
			Model:       MxbaiEmbedLargeV1,
			Dim:         1024,
			Description: "MixedBread Embed Large v1",
			ModelCode:   "mixedbread-ai/mxbai-embed-large-v1",
			ModelFile:   "onnx/model.onnx",
		},
		MxbaiEmbedLargeV1Q: {
			Model:       MxbaiEmbedLargeV1Q,
			Dim:         1024,
			Description: "MixedBread Embed Large v1 (Quantized)",
			ModelCode:   "mixedbread-ai/mxbai-embed-large-v1",
			ModelFile:   "onnx/model_quantized.onnx",
		},

		// GTE models
		GTEBaseENV15: {
			Model:       GTEBaseENV15,
			Dim:         768,
			Description: "GTE Base English v1.5",
			ModelCode:   "Alibaba-NLP/gte-base-en-v1.5",
			ModelFile:   "onnx/model.onnx",
		},
		GTEBaseENV15Q: {
			Model:       GTEBaseENV15Q,
			Dim:         768,
			Description: "GTE Base English v1.5 (Quantized)",
			ModelCode:   "Alibaba-NLP/gte-base-en-v1.5",
			ModelFile:   "onnx/model_quantized.onnx",
		},
		GTELargeENV15: {
			Model:       GTELargeENV15,
			Dim:         1024,
			Description: "GTE Large English v1.5",
			ModelCode:   "Alibaba-NLP/gte-large-en-v1.5",
			ModelFile:   "onnx/model.onnx",
		},

		// Paraphrase models
		ParaphraseMLMiniLML12V2: {
			Model:       ParaphraseMLMiniLML12V2,
			Dim:         384,
			Description: "Paraphrase Multilingual MiniLM-L12-v2",
			ModelCode:   "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx",
			ModelFile:   "model.onnx",
		},
		ParaphraseMLMiniLML12V2Q: {
			Model:       ParaphraseMLMiniLML12V2Q,
			Dim:         384,
			Description: "Paraphrase Multilingual MiniLM-L12-v2 (Quantized)",
			ModelCode:   "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
			ModelFile:   "model_optimized.onnx",
		},

		// Jina models
		JinaEmbeddingsV2BaseCode: {
			Model:          JinaEmbeddingsV2BaseCode,
			Dim:            768,
			Description:    "Jina Embeddings v2 Base for Code",
			ModelCode:      "jinaai/jina-embeddings-v2-base-code",
			ModelFile:      "onnx/model.onnx",
			NoTokenTypeIDs: true,
		},
	}

	registry.RegisterAll(models)
}

// GetModelInfo returns info for a model, or nil if not found.
func GetModelInfo(m Model) *model.Info[Model] {
	return registry.Get(m)
}

// ListModels returns all supported model infos.
func ListModels() []*model.Info[Model] {
	return registry.List()
}

// DefaultPooling returns the default pooling strategy for a model.
func (m Model) DefaultPooling() pooling.Strategy {
	switch m {
	// CLS pooling models
	case AllMiniLML6V2, AllMiniLML6V2Q, AllMiniLML12V2, // MiniLM uses CLS for canonical values compatibility
		BGESmallENV15, BGESmallENV15Q, BGEBaseENV15, BGEBaseENV15Q,
		BGELargeENV15, BGELargeENV15Q, BGESmallEN, BGEBaseEN, BGESmallZH,
		GTEBaseENV15, GTEBaseENV15Q, GTELargeENV15,
		MxbaiEmbedLargeV1, MxbaiEmbedLargeV1Q:
		return pooling.Cls
	// Mean pooling models (default)
	default:
		return pooling.Mean
	}
}

// Quantization returns the quantization mode for a model.
func (m Model) Quantization() quantization.Mode {
	switch m {
	// Dynamic quantization models
	case AllMiniLML6V2Q, NomicEmbedTextV15Q,
		MxbaiEmbedLargeV1Q, GTEBaseENV15Q:
		return quantization.Dynamic
	// Static quantization models
	case BGESmallENV15Q, BGEBaseENV15Q, BGELargeENV15Q,
		ParaphraseMLMiniLML12V2Q:
		return quantization.Static
	// No quantization (default)
	default:
		return quantization.None
	}
}

// String returns the string representation of the model.
func (m Model) String() string {
	return string(m)
}
