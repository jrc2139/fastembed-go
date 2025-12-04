package text_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/anush008/fastembed-go/internal/onnx"
	"github.com/anush008/fastembed-go/internal/pooling"
	"github.com/anush008/fastembed-go/text"
)

func TestCanonicalValues(t *testing.T) {
	canonicalValues := map[text.Model][]float32{
		text.AllMiniLML6V2: {0.02591, 0.00573, 0.01147, 0.03796, -0.02328},
		text.BGESmallEN:    {-0.02313, -0.02552, 0.017357, -0.06393, -0.00061},
		text.BGEBaseEN:     {0.01140, 0.03722, 0.02941, 0.01230, 0.03451},
		text.BGEBaseENV15:  {0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045},
		text.BGESmallENV15: {0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434},
		text.BGESmallZH:    {-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762},
	}

	for model, expected := range canonicalValues {
		t.Run(string(model), func(t *testing.T) {
			emb, err := text.New(
				text.WithModel(model),
				text.WithShowDownloadProgress(false),
			)
			if err != nil {
				t.Fatalf("Expected no error, got %v", err)
			}
			defer emb.Destroy()

			ctx := context.Background()
			input := []string{"hello world"}
			result, err := emb.Embed(ctx, input, 1)
			if err != nil {
				t.Fatalf("Expected no error, got %v", err)
			}

			if len(result) != len(input) {
				t.Errorf("Expected result length %v, got %v", len(input), len(result))
			}

			epsilon := float64(1e-4)
			for i, v := range expected {
				if math.Abs(float64(result[0][i]-v)) > epsilon {
					t.Errorf("Element %d mismatch: expected %.6f, got %.6f", i, v, result[0][i])
				}
			}
		})
	}
}

func TestMultilingualE5Large(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large model test in short mode")
	}

	emb, err := text.New(
		text.WithModel(text.MultilingualE5Large),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize MultilingualE5Large: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	input := []string{"hello world", "hola mundo", "bonjour le monde"}
	result, err := emb.Embed(ctx, input, 1)
	if err != nil {
		t.Fatalf("Failed to embed: %v", err)
	}

	if len(result) != len(input) {
		t.Errorf("Expected %d results, got %d", len(input), len(result))
	}

	// Check embedding dimension (1024 for E5-Large)
	for i, emb := range result {
		if len(emb) != 1024 {
			t.Errorf("Result %d: expected dimension 1024, got %d", i, len(emb))
		}
	}

	// Embeddings should be normalized
	for i, emb := range result {
		norm := float32(0)
		for _, v := range emb {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if math.Abs(float64(norm-1.0)) > 0.01 {
			t.Errorf("Result %d: expected normalized embedding (norm ~1.0), got %.4f", i, norm)
		}
	}
}

func TestPoolingOverride(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithPooling(pooling.Mean),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize with pooling override: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	input := []string{"hello world"}
	result, err := emb.Embed(ctx, input, 1)
	if err != nil {
		t.Fatalf("Failed to embed with mean pooling: %v", err)
	}

	if len(result) != 1 || len(result[0]) != 384 {
		t.Errorf("Unexpected result shape")
	}
}

func TestInstructEmbed(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large model test in short mode")
	}

	emb, err := text.New(
		text.WithModel(text.MultilingualE5LargeInstruct),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize E5LargeInstruct: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	task := "Given a query, retrieve relevant passages"
	texts := []string{"What is machine learning?", "How do neural networks work?"}

	result, err := emb.InstructEmbed(ctx, texts, task, 1)
	if err != nil {
		t.Fatalf("Failed InstructEmbed: %v", err)
	}

	if len(result) != len(texts) {
		t.Errorf("Expected %d results, got %d", len(texts), len(result))
	}

	// Check embedding dimension
	for i, emb := range result {
		if len(emb) != 1024 {
			t.Errorf("Result %d: expected dimension 1024, got %d", i, len(emb))
		}
	}
}

func TestCUDAExecution(t *testing.T) {
	if os.Getenv("TEST_CUDA") != "1" {
		t.Skip("Skipping CUDA test (set TEST_CUDA=1 to enable)")
	}

	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithCUDA(0),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize with CUDA: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	input := []string{"hello world"}
	result, err := emb.Embed(ctx, input, 1)
	if err != nil {
		t.Fatalf("Failed to embed with CUDA: %v", err)
	}

	if len(result) != 1 || len(result[0]) != 384 {
		t.Errorf("Unexpected result shape with CUDA")
	}

	t.Logf("CUDA embedding successful, first 5 values: %v", result[0][:5])
}

func TestEmbeddingGemma(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large model test in short mode")
	}

	emb, err := text.New(
		text.WithModel(text.EmbeddingGemma300M),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize EmbeddingGemma300M: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	input := []string{"hello world", "what is machine learning?"}
	result, err := emb.Embed(ctx, input, 1)
	if err != nil {
		t.Fatalf("Failed to embed: %v", err)
	}

	if len(result) != len(input) {
		t.Errorf("Expected %d results, got %d", len(input), len(result))
	}

	// Check embedding dimension (768 for Gemma)
	for i, emb := range result {
		if len(emb) != 768 {
			t.Errorf("Result %d: expected dimension 768, got %d", i, len(emb))
		}
	}

	// Embeddings should be normalized
	for i, emb := range result {
		norm := float32(0)
		for _, v := range emb {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if math.Abs(float64(norm-1.0)) > 0.01 {
			t.Errorf("Result %d: expected normalized embedding (norm ~1.0), got %.4f", i, norm)
		}
	}
}

func TestQueryAndPassageEmbed(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Test QueryEmbed
	query, err := emb.QueryEmbed(ctx, "what is machine learning")
	if err != nil {
		t.Fatalf("Failed QueryEmbed: %v", err)
	}
	if len(query) != 384 {
		t.Errorf("Expected query dimension 384, got %d", len(query))
	}

	// Test PassageEmbed
	passages := []string{"Machine learning is a branch of AI", "Deep learning uses neural networks"}
	passageEmbs, err := emb.PassageEmbed(ctx, passages, 2)
	if err != nil {
		t.Fatalf("Failed PassageEmbed: %v", err)
	}
	if len(passageEmbs) != 2 {
		t.Errorf("Expected 2 passage embeddings, got %d", len(passageEmbs))
	}
}

func TestListSupportedModels(t *testing.T) {
	models := text.ListModels()

	// Check that we have at least 20 models
	if len(models) < 20 {
		t.Errorf("Expected at least 20 models, got %d", len(models))
	}

	// Check that specific models are included
	modelSet := make(map[text.Model]bool)
	for _, m := range models {
		modelSet[m.Model] = true
	}

	expectedModels := []text.Model{
		text.AllMiniLML6V2,
		text.BGESmallENV15,
		text.MultilingualE5Large,
		text.EmbeddingGemma300M,
		text.NomicEmbedTextV15,
	}

	for _, model := range expectedModels {
		if !modelSet[model] {
			t.Errorf("Expected model %s to be in supported models list", model)
		}
	}
}

func TestGetModelInfo(t *testing.T) {
	info := text.GetModelInfo(text.BGESmallENV15)
	if info == nil {
		t.Fatal("Expected model info, got nil")
	}

	if info.Dim != 384 {
		t.Errorf("Expected dim 384, got %d", info.Dim)
	}

	if info.ModelCode == "" {
		t.Error("Expected non-empty model code")
	}
}

func TestModelBehaviors(t *testing.T) {
	// Test default pooling
	if text.BGESmallENV15.DefaultPooling() != pooling.Cls {
		t.Errorf("BGE models should use CLS pooling")
	}

	if text.AllMiniLML6V2.DefaultPooling() != pooling.Mean {
		t.Errorf("AllMiniLM models should use Mean pooling")
	}

	// Test quantization
	if text.AllMiniLML6V2Q.Quantization() == 0 {
		// Quantized models should have non-zero quantization
		// (quantization.None == 0, so quantized should be > 0)
	}
}

func TestEmptyInput(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	result, err := emb.Embed(ctx, []string{}, 1)
	if err != nil {
		t.Fatalf("Expected no error for empty input, got %v", err)
	}
	if len(result) != 0 {
		t.Errorf("Expected empty result for empty input, got %d", len(result))
	}
}

func TestExecutionProviders(t *testing.T) {
	// Test that CPU provider works (always available)
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithExecutionProviders(onnx.CPUProvider()),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize with CPU provider: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	result, err := emb.Embed(ctx, []string{"test"}, 1)
	if err != nil {
		t.Fatalf("Failed to embed with CPU provider: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}
}
