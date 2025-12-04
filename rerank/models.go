package rerank

// Model represents a supported reranker model.
type Model string

const (
	// BGERerankerBase is the default reranker for English and Chinese.
	BGERerankerBase Model = "BAAI/bge-reranker-base"

	// BGERerankerV2M3 is a multilingual reranker.
	BGERerankerV2M3 Model = "rozgo/bge-reranker-v2-m3"

	// JINARerankerV1TurboEn is a fast English reranker.
	JINARerankerV1TurboEn Model = "jinaai/jina-reranker-v1-turbo-en"

	// JINARerankerV2BaseMultilingual is a multilingual reranker.
	JINARerankerV2BaseMultilingual Model = "jinaai/jina-reranker-v2-base-multilingual"
)

// ModelInfo contains metadata about a reranker model.
type ModelInfo struct {
	Model           Model
	Description     string
	ModelCode       string
	ModelFile       string
	AdditionalFiles []string
}

var rerankerModels = []ModelInfo{
	{
		Model:           BGERerankerBase,
		Description:     "BGE reranker for English and Chinese",
		ModelCode:       "BAAI/bge-reranker-base",
		ModelFile:       "onnx/model.onnx",
		AdditionalFiles: nil,
	},
	{
		Model:           BGERerankerV2M3,
		Description:     "BGE reranker v2 with multilingual support",
		ModelCode:       "rozgo/bge-reranker-v2-m3",
		ModelFile:       "model.onnx",
		AdditionalFiles: []string{"model.onnx.data"},
	},
	{
		Model:           JINARerankerV1TurboEn,
		Description:     "JINA reranker v1 turbo for English",
		ModelCode:       "jinaai/jina-reranker-v1-turbo-en",
		ModelFile:       "onnx/model.onnx",
		AdditionalFiles: nil,
	},
	{
		Model:           JINARerankerV2BaseMultilingual,
		Description:     "JINA reranker v2 base multilingual",
		ModelCode:       "jinaai/jina-reranker-v2-base-multilingual",
		ModelFile:       "onnx/model.onnx",
		AdditionalFiles: nil,
	},
}

// ListSupportedModels returns all supported reranker models.
func ListSupportedModels() []ModelInfo {
	return rerankerModels
}

// GetModelInfo returns metadata for a specific model.
func GetModelInfo(model Model) *ModelInfo {
	for _, m := range rerankerModels {
		if m.Model == model {
			return &m
		}
	}
	return nil
}
