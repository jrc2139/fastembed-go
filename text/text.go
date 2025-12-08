package text

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"path/filepath"

	"github.com/alitto/pond/v2"
	tk "github.com/sugarme/tokenizer"

	"github.com/anush008/fastembed-go/cuda"
	"github.com/anush008/fastembed-go/internal/download"
	"github.com/anush008/fastembed-go/internal/onnx"
	"github.com/anush008/fastembed-go/internal/output"
	"github.com/anush008/fastembed-go/internal/pooling"
	"github.com/anush008/fastembed-go/internal/quantization"
	"github.com/anush008/fastembed-go/internal/tokenizer"
)

// Embedding is a dense vector of float32 values.
type Embedding = []float32

// TextEmbedding provides dense text embedding generation.
type TextEmbedding struct {
	tokenizer     *tokenizer.Tokenizer
	session       *onnx.Session
	pooling       pooling.Strategy
	quantization  quantization.Mode
	outputKey     string
	dim           int
	useTokenTypes bool
	is2DOutput    bool // true for models with direct sentence embeddings (e.g., OutputKey="sentence_embedding")
	logger        *slog.Logger
	pool          pond.Pool // Worker pool for parallel batch processing
}

// New creates a TextEmbedding from a built-in model.
func New(opts ...Option) (*TextEmbedding, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	log := cfg.Logger

	info := GetModelInfo(cfg.Model)
	if info == nil {
		return nil, fmt.Errorf("model %s not found in registry", cfg.Model)
	}

	log.Info("initializing text embedding model",
		slog.String("model", string(cfg.Model)),
		slog.String("model_code", info.ModelCode),
		slog.Int("dimension", info.Dim),
	)

	// Download or retrieve cached model
	modelDir, err := download.RetrieveModel(download.Config{
		ModelCode:       info.ModelCode,
		ModelFile:       info.ModelFile,
		AdditionalFiles: info.AdditionalFiles,
		TokenizerPath:   info.TokenizerPath,
		CacheDir:        cfg.CacheDir,
		ShowProgress:    cfg.ShowProgress,
		Logger:          log,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve model: %w", err)
	}

	// Load tokenizer
	log.Debug("loading tokenizer", slog.String("path", modelDir))
	tknzer, err := tokenizer.LoadFromPath(modelDir, cfg.MaxLength)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Create ONNX session
	modelPath := filepath.Join(modelDir, filepath.Base(info.ModelFile))
	log.Debug("creating ONNX session",
		slog.String("model_path", modelPath),
		slog.Int("provider_count", len(cfg.Providers)),
	)
	session, err := onnx.NewSession(modelPath, cfg.Providers...)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	// Determine pooling
	p := cfg.Model.DefaultPooling()
	if cfg.Pooling != nil {
		p = *cfg.Pooling
	}

	// Determine output key and whether output is 2D
	// Models with custom OutputKey (like "sentence_embedding") produce direct sentence embeddings (2D)
	// Models with default "last_hidden_state" produce token embeddings (3D) that need pooling
	outputKey := "last_hidden_state"
	is2DOutput := false
	if info.OutputKey != nil && info.OutputKey.Type == output.ByName {
		outputKey = info.OutputKey.Name
		is2DOutput = true // Custom output key indicates 2D sentence embeddings
	}

	// Auto-tune batch size and workers based on available VRAM
	if cfg.AutoTune {
		modelName := modelNameForProfile(cfg.Model)
		params, err := cuda.AutoTune(cfg.CUDADeviceID, modelName, log)
		if err != nil {
			log.Warn("auto-tune failed, using defaults", "error", err)
		} else {
			cfg.MaxWorkers = params.MaxWorkers
			log.Info("auto-tuned parameters",
				"batch_size", params.BatchSize,
				"max_workers", params.MaxWorkers,
			)
		}
	}

	// Create worker pool for parallel batch processing
	workerPool := pond.NewPool(cfg.MaxWorkers)

	log.Info("text embedding model ready",
		slog.String("model", string(cfg.Model)),
		slog.String("pooling", p.String()),
		slog.String("output_key", outputKey),
		slog.Int("max_workers", cfg.MaxWorkers),
	)

	return &TextEmbedding{
		tokenizer:     tknzer,
		session:       session,
		pooling:       p,
		quantization:  cfg.Model.Quantization(),
		outputKey:     outputKey,
		dim:           info.Dim,
		useTokenTypes: !info.NoTokenTypeIDs,
		is2DOutput:    is2DOutput,
		logger:        log,
		pool:          workerPool,
	}, nil
}

// NewFromUserDefined creates a TextEmbedding from a user-provided model.
func NewFromUserDefined(model *UserDefinedModel, opts ...Option) (*TextEmbedding, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Load tokenizer from bytes
	tknzer, err := tokenizer.LoadFromBytes(model.TokenizerFiles, cfg.MaxLength)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Note: BYOM requires both tokenizer and ONNX session from memory.
	// The tokenizer.LoadFromBytes call above will fail with "not yet implemented"
	// so we won't reach here. When BYOM is fully implemented, this will create
	// the ONNX session and return a TextEmbedding instance.
	_ = tknzer // Will be used when BYOM is implemented
	return nil, fmt.Errorf("NewFromUserDefined: not yet fully implemented")
}

// Embed generates embeddings for the given texts.
// batchSize controls parallel processing (0 = default of 256).
func (t *TextEmbedding) Embed(ctx context.Context, texts []string, batchSize int) ([]Embedding, error) {
	if len(texts) == 0 {
		return []Embedding{}, nil
	}

	// Validate batch size for quantization mode
	if err := t.quantization.ValidateBatchSize(batchSize, len(texts)); err != nil {
		return nil, err
	}

	// Adjust batch size - default 64 is optimal for parallel processing
	// Smaller batches (64-128) with worker parallelization outperform large batches
	batchSize = t.quantization.AdjustBatchSize(batchSize, len(texts), 64)
	numBatches := (len(texts) + batchSize - 1) / batchSize

	t.logger.Debug("embedding texts",
		slog.Int("text_count", len(texts)),
		slog.Int("batch_size", batchSize),
		slog.Int("num_batches", numBatches),
	)

	// Process in batches using worker pool
	embeddings := make([]Embedding, len(texts))
	group := t.pool.NewGroup()

	for i := 0; i < len(texts); i += batchSize {
		start, end := i, i+batchSize
		if end > len(texts) {
			end = len(texts)
		}

		group.SubmitErr(func() error {
			// Check context cancellation
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			batch := texts[start:end]
			batchEmbeddings, err := t.embedBatch(batch)
			if err != nil {
				return err
			}

			for j, emb := range batchEmbeddings {
				embeddings[start+j] = emb
			}
			return nil
		})
	}

	// Wait for all tasks to complete
	if err := group.Wait(); err != nil {
		return nil, err
	}

	return embeddings, nil
}

// embedBatch embeds a single batch of texts.
func (t *TextEmbedding) embedBatch(texts []string) ([]Embedding, error) {
	// Tokenize
	inputs := make([]tk.EncodeInput, len(texts))
	for i, text := range texts {
		seq := tk.NewInputSequence(text)
		inputs[i] = tk.NewSingleEncodeInput(seq)
	}

	encodings, err := t.tokenizer.EncodeBatch(inputs, true)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	// Flatten token data with direct indexing (faster than append)
	batchSize := len(inputs)
	seqLen := encodings[0].Len()
	totalLen := batchSize * seqLen

	inputIDs := make([]int64, totalLen)
	attentionMask := make([]int64, totalLen)
	tokenTypeIDs := make([]int64, totalLen)

	for i, enc := range encodings {
		baseIdx := i * seqLen
		ids := enc.GetIds()
		masks := enc.GetAttentionMask()
		types := enc.GetTypeIds()

		for j := 0; j < seqLen; j++ {
			idx := baseIdx + j
			inputIDs[idx] = int64(ids[j])
			attentionMask[idx] = int64(masks[j])
			if j < len(types) {
				tokenTypeIDs[idx] = int64(types[j])
			}
		}
	}

	// Run ONNX inference
	outputData, outputShape, err := onnx.RunTextEmbedding(
		t.session,
		inputIDs, attentionMask, tokenTypeIDs,
		batchSize, seqLen, t.dim,
		t.outputKey,
		t.useTokenTypes,
		t.is2DOutput,
	)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Process output based on shape
	var embeddings []Embedding
	if len(outputShape) == 2 {
		// 2D output: direct sentence embeddings (zero-copy sub-slicing)
		embeddings = extract2DZeroCopy(outputData, batchSize, t.dim)
	} else {
		// 3D output: apply pooling
		embeddings = t.pooling.Pool(outputData, batchSize, seqLen, t.dim, attentionMask)
	}

	// Normalize embeddings in-place
	for i := range embeddings {
		normalizeInPlace(embeddings[i])
	}

	return embeddings, nil
}

// QueryEmbed embeds a single query with "query: " prefix (for retrieval).
func (t *TextEmbedding) QueryEmbed(ctx context.Context, query string) (Embedding, error) {
	embeddings, err := t.Embed(ctx, []string{"query: " + query}, 1)
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

// PassageEmbed embeds passages with "passage: " prefix (for retrieval).
func (t *TextEmbedding) PassageEmbed(ctx context.Context, passages []string, batchSize int) ([]Embedding, error) {
	prefixed := make([]string, len(passages))
	for i, p := range passages {
		prefixed[i] = "passage: " + p
	}
	return t.Embed(ctx, prefixed, batchSize)
}

// InstructEmbed embeds texts with instruction prefix (for E5-Instruct models).
// Format: "Instruct: {task}\nQuery: {text}"
func (t *TextEmbedding) InstructEmbed(ctx context.Context, texts []string, task string, batchSize int) ([]Embedding, error) {
	prefixed := make([]string, len(texts))
	for i, text := range texts {
		prefixed[i] = fmt.Sprintf("Instruct: %s\nQuery: %s", task, text)
	}
	return t.Embed(ctx, prefixed, batchSize)
}

// Destroy releases resources held by the embedding model.
func (t *TextEmbedding) Destroy() error {
	// Stop the worker pool
	if t.pool != nil {
		t.pool.StopAndWait()
	}

	if t.session != nil {
		return t.session.Destroy()
	}
	return nil
}

// TokenCount returns the number of tokens in the given text.
// This is useful for chunking text to fit within model limits.
func (t *TextEmbedding) TokenCount(text string) (int, error) {
	seq := tk.NewInputSequence(text)
	input := tk.NewSingleEncodeInput(seq)
	encodings, err := t.tokenizer.EncodeBatch([]tk.EncodeInput{input}, false)
	if err != nil {
		return 0, fmt.Errorf("failed to encode text: %w", err)
	}
	if len(encodings) == 0 {
		return 0, nil
	}
	return len(encodings[0].GetIds()), nil
}

// MaxTokens returns the maximum token length for this model.
func (t *TextEmbedding) MaxTokens() int {
	return t.tokenizer.MaxLength()
}

// Dimension returns the embedding dimension for this model.
func (t *TextEmbedding) Dimension() int {
	return t.dim
}

// extract2D extracts embeddings from 2D output [batch, dim] with copy.
// Used when the output tensor may be reused and we need independent copies.
func extract2D(data []float32, batchSize, dim int) []Embedding {
	embeddings := make([]Embedding, batchSize)
	for i := 0; i < batchSize; i++ {
		embedding := make([]float32, dim)
		copy(embedding, data[i*dim:(i+1)*dim])
		embeddings[i] = embedding
	}
	return embeddings
}

// extract2DZeroCopy extracts embeddings from 2D output [batch, dim] without copying.
// Creates sub-slices pointing to the original data.
// IMPORTANT: The returned embeddings share memory with the input data.
// This is safe because ONNX output tensor is destroyed after this function returns,
// but we modify the embeddings in-place (normalization), so the data remains valid.
// We still need to copy because the underlying tensor will be destroyed.
// However, we can optimize by doing a single allocation.
func extract2DZeroCopy(data []float32, batchSize, dim int) []Embedding {
	// Allocate all embedding memory in a single allocation
	allEmbeddings := make([]float32, batchSize*dim)
	copy(allEmbeddings, data)

	// Create slice headers pointing into the single allocation
	embeddings := make([]Embedding, batchSize)
	for i := 0; i < batchSize; i++ {
		embeddings[i] = allEmbeddings[i*dim : (i+1)*dim]
	}
	return embeddings
}

// normalize normalizes a vector to unit length (L2 normalization).
// Returns a new normalized vector.
func normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	norm := float32(1.0 / math.Sqrt(sum))
	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = x * norm
	}
	return result
}

// normalizeInPlace normalizes a vector to unit length (L2 normalization) in place.
// Modifies the input vector directly, avoiding allocation.
func normalizeInPlace(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	norm := float32(1.0 / math.Sqrt(sum))
	for i, x := range v {
		v[i] = x * norm
	}
}

// modelNameForProfile maps Model to the profile name used in cuda.AutoTune.
func modelNameForProfile(m Model) string {
	switch m {
	case BGESmallENV15, BGESmallENV15Q:
		return "bge-small-en-v1.5"
	case BGEBaseENV15, BGEBaseENV15Q:
		return "bge-base-en-v1.5"
	case BGELargeENV15, BGELargeENV15Q:
		return "bge-large-en-v1.5"
	case AllMiniLML6V2, AllMiniLML6V2Q:
		return "all-MiniLM-L6-v2"
	case AllMiniLML12V2:
		return "all-MiniLM-L6-v2" // Similar profile
	case MultilingualE5Small:
		return "multilingual-e5-small"
	case MultilingualE5Base:
		return "multilingual-e5-base"
	case MultilingualE5Large, MultilingualE5LargeInstruct:
		return "multilingual-e5-large"
	case NomicEmbedTextV1:
		return "nomic-embed-text-v1"
	case NomicEmbedTextV15, NomicEmbedTextV15Q:
		return "nomic-embed-text-v1.5"
	case MxbaiEmbedLargeV1, MxbaiEmbedLargeV1Q:
		return "mxbai-embed-large-v1"
	case GTEBaseENV15, GTEBaseENV15Q:
		return "gte-large-en-v1.5" // Use large profile for safety
	case GTELargeENV15:
		return "gte-large-en-v1.5"
	case JinaEmbeddingsV2BaseCode:
		return "jina-embeddings-v2-base-code"
	case EmbeddingGemma300M, EmbeddingGemma300MQ4:
		return "embedding-gemma-300m"
	case ParaphraseMLMiniLML12V2, ParaphraseMLMiniLML12V2Q:
		return "paraphrase-MiniLM-L12-v2"
	default:
		return string(m)
	}
}
