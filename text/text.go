package text

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"path/filepath"
	"sync"

	tk "github.com/sugarme/tokenizer"

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

	log.Info("text embedding model ready",
		slog.String("model", string(cfg.Model)),
		slog.String("pooling", p.String()),
		slog.String("output_key", outputKey),
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

	// Adjust batch size
	batchSize = t.quantization.AdjustBatchSize(batchSize, len(texts), 256)
	numBatches := (len(texts) + batchSize - 1) / batchSize

	t.logger.Debug("embedding texts",
		slog.Int("text_count", len(texts)),
		slog.Int("batch_size", batchSize),
		slog.Int("num_batches", numBatches),
	)

	// Process in batches
	embeddings := make([]Embedding, len(texts))
	var wg sync.WaitGroup
	errCh := make(chan error, numBatches)

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			// Check context cancellation
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			default:
			}

			batch := texts[start:end]
			batchEmbeddings, err := t.embedBatch(batch)
			if err != nil {
				errCh <- err
				return
			}

			for j, emb := range batchEmbeddings {
				embeddings[start+j] = emb
			}
		}(i, end)
	}

	wg.Wait()
	close(errCh)

	// Return first error if any
	for err := range errCh {
		if err != nil {
			return nil, err
		}
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

	// Flatten token data
	batchSize := len(inputs)
	seqLen := encodings[0].Len()

	inputIDs := make([]int64, 0, batchSize*seqLen)
	attentionMask := make([]int64, 0, batchSize*seqLen)
	tokenTypeIDs := make([]int64, 0, batchSize*seqLen)

	for _, enc := range encodings {
		for _, id := range enc.GetIds() {
			inputIDs = append(inputIDs, int64(id))
		}
		for _, mask := range enc.GetAttentionMask() {
			attentionMask = append(attentionMask, int64(mask))
		}
		for _, typeID := range enc.GetTypeIds() {
			tokenTypeIDs = append(tokenTypeIDs, int64(typeID))
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
		// 2D output: direct sentence embeddings
		embeddings = extract2D(outputData, batchSize, t.dim)
	} else {
		// 3D output: apply pooling
		embeddings = t.pooling.Pool(outputData, batchSize, seqLen, t.dim, attentionMask)
	}

	// Normalize embeddings
	for i := range embeddings {
		embeddings[i] = normalize(embeddings[i])
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
	if t.session != nil {
		return t.session.Destroy()
	}
	return nil
}

// extract2D extracts embeddings from 2D output [batch, dim].
func extract2D(data []float32, batchSize, dim int) []Embedding {
	embeddings := make([]Embedding, batchSize)
	for i := 0; i < batchSize; i++ {
		embedding := make([]float32, dim)
		copy(embedding, data[i*dim:(i+1)*dim])
		embeddings[i] = embedding
	}
	return embeddings
}

// normalize normalizes a vector to unit length (L2 normalization).
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
