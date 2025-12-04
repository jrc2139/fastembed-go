// Package rerank provides cross-encoder reranking functionality using ONNX models.
package rerank

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"

	tk "github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"

	"github.com/anush008/fastembed-go/internal/download"
	"github.com/anush008/fastembed-go/internal/onnx"
)

// Result represents a reranking result for a single document.
type Result struct {
	Document string  // The original document text (if ReturnDocuments=true)
	Score    float32 // Relevance score (higher = more relevant)
	Index    int     // Original index in the input documents array
}

// TextRerank is a cross-encoder reranker.
type TextRerank struct {
	session      *onnx.Session
	tokenizer    *tk.Tokenizer
	maxLength    int
	needTypeIDs  bool
	model        Model
	showProgress bool
	cacheDir     string
	useCUDA      bool
	cudaDeviceID int
	logger       *slog.Logger
}

// defaultLogger returns a default JSON logger to stderr at INFO level.
func defaultLogger() *slog.Logger {
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
}

// Option configures a TextRerank.
type Option func(*TextRerank)

// WithModel sets the reranker model.
func WithModel(m Model) Option {
	return func(r *TextRerank) {
		r.model = m
	}
}

// WithMaxLength sets the maximum sequence length (default: 512).
func WithMaxLength(n int) Option {
	return func(r *TextRerank) {
		r.maxLength = n
	}
}

// WithCacheDir sets the model cache directory.
func WithCacheDir(dir string) Option {
	return func(r *TextRerank) {
		r.cacheDir = dir
	}
}

// WithShowDownloadProgress enables download progress display.
func WithShowDownloadProgress(show bool) Option {
	return func(r *TextRerank) {
		r.showProgress = show
	}
}

// WithCUDA enables CUDA execution provider.
func WithCUDA(deviceID int) Option {
	return func(r *TextRerank) {
		r.useCUDA = true
		r.cudaDeviceID = deviceID
	}
}

// WithLogger sets a custom slog.Logger for the reranker.
// If not set, a no-op logger is used (no logging output).
func WithLogger(logger *slog.Logger) Option {
	return func(r *TextRerank) {
		r.logger = logger
	}
}

// New creates a new TextRerank instance.
func New(opts ...Option) (*TextRerank, error) {
	r := &TextRerank{
		model:     BGERerankerBase,
		maxLength: 512,
		logger:    defaultLogger(),
	}

	for _, opt := range opts {
		opt(r)
	}

	log := r.logger

	// Get model info
	info := GetModelInfo(r.model)
	if info == nil {
		return nil, fmt.Errorf("unknown reranker model: %s", r.model)
	}

	log.Info("initializing reranker",
		slog.String("model", string(r.model)),
		slog.String("model_code", info.ModelCode),
	)

	// Determine cache directory
	cacheDir := r.cacheDir
	if cacheDir == "" {
		cacheDir = download.GetCacheDir()
	}

	// Download model
	cfg := download.Config{
		ModelCode:       info.ModelCode,
		ModelFile:       info.ModelFile,
		AdditionalFiles: info.AdditionalFiles,
		CacheDir:        cacheDir,
		ShowProgress:    r.showProgress,
		Logger:          log,
	}
	modelDir, err := download.RetrieveModel(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve model: %w", err)
	}

	// Load tokenizer
	log.Debug("loading tokenizer", slog.String("path", modelDir))
	configFile := filepath.Join(modelDir, "tokenizer.json")
	tokenizer, err := pretrained.FromFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Configure truncation
	tokenizer.WithTruncation(&tk.TruncationParams{
		MaxLength: r.maxLength,
		Strategy:  tk.LongestFirst,
	})

	// Configure padding
	padID, ok := tokenizer.TokenToId("[PAD]")
	if !ok {
		padID = 0
	}
	tokenizer.WithPadding(&tk.PaddingParams{
		Strategy:  *tk.NewPaddingStrategy(),
		PadId:     int(padID),
		PadToken:  "[PAD]",
		Direction: tk.Right,
	})

	// Create ONNX session with execution providers
	// Download saves files with basename only, so use basename for path
	modelPath := filepath.Join(modelDir, filepath.Base(info.ModelFile))
	var providers []onnx.ExecutionProvider
	if r.useCUDA {
		providers = append(providers, onnx.CUDAProvider(r.cudaDeviceID))
	}

	log.Debug("creating ONNX session",
		slog.String("model_path", modelPath),
		slog.Int("provider_count", len(providers)),
	)
	session, err := onnx.NewSession(modelPath, providers...)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	// Check if model needs token_type_ids
	needTypeIDs := session.HasInput("token_type_ids")

	r.session = session
	r.tokenizer = tokenizer
	r.needTypeIDs = needTypeIDs

	log.Info("reranker ready",
		slog.String("model", string(r.model)),
		slog.Bool("needs_token_type_ids", needTypeIDs),
	)

	return r, nil
}

// Rerank scores and reorders documents by relevance to a query.
func (r *TextRerank) Rerank(query string, documents []string, returnDocuments bool) ([]Result, error) {
	return r.RerankWithBatchSize(query, documents, returnDocuments, 256)
}

// RerankWithBatchSize scores documents with a custom batch size.
func (r *TextRerank) RerankWithBatchSize(query string, documents []string, returnDocuments bool, batchSize int) ([]Result, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	numBatches := (len(documents) + batchSize - 1) / batchSize
	r.logger.Debug("reranking documents",
		slog.Int("document_count", len(documents)),
		slog.Int("batch_size", batchSize),
		slog.Int("num_batches", numBatches),
	)

	results := make([]Result, 0, len(documents))

	// Process in batches
	for start := 0; start < len(documents); start += batchSize {
		end := start + batchSize
		if end > len(documents) {
			end = len(documents)
		}
		batch := documents[start:end]

		scores, err := r.scoreBatch(query, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to score batch: %w", err)
		}

		for i, score := range scores {
			result := Result{
				Score: score,
				Index: start + i,
			}
			if returnDocuments {
				result.Document = batch[i]
			}
			results = append(results, result)
		}
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// scoreBatch scores a batch of documents against the query.
func (r *TextRerank) scoreBatch(query string, documents []string) ([]float32, error) {
	// Tokenize query-document pairs
	inputs := make([]tk.EncodeInput, len(documents))
	querySeq := tk.NewInputSequence(query)

	for i, doc := range documents {
		docSeq := tk.NewInputSequence(doc)
		inputs[i] = tk.NewDualEncodeInput(querySeq, docSeq)
	}

	encodings, err := r.tokenizer.EncodeBatch(inputs, true)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	// Flatten token data
	batchSize := len(documents)
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
	scores, err := onnx.RunReranking(
		r.session,
		inputIDs, attentionMask, tokenTypeIDs,
		batchSize, seqLen,
		r.needTypeIDs,
	)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	return scores, nil
}

// Destroy releases resources.
func (r *TextRerank) Destroy() error {
	if r.session != nil {
		return r.session.Destroy()
	}
	return nil
}
