package fastembed_test

import (
	"math"
	"os"
	"testing"

	fastembed "github.com/anush008/fastembed-go"
)

func TestCanonicalValues(t *testing.T) {
	canonicalValues := map[fastembed.EmbeddingModel]([]float32){
		fastembed.AllMiniLML6V2: []float32{0.02591, 0.00573, 0.01147, 0.03796, -0.02328},
		fastembed.BGESmallEN:    []float32{-0.02313, -0.02552, 0.017357, -0.06393, -0.00061},
		fastembed.BGEBaseEN:     []float32{0.01140, 0.03722, 0.02941, 0.01230, 0.03451},
		fastembed.BGEBaseENV15:  []float32{0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045},
		fastembed.BGESmallENV15: []float32{0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434},
		fastembed.BGESmallZH:    []float32{-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762},
	}

	for model, expected := range canonicalValues {
		fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
			Model: model,
		})
		defer fe.Destroy()
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		input := []string{"hello world"}
		result, err := fe.Embed(input, 1)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if len(result) != len(input) {
			t.Errorf("Expected result length %v, got %v", len(input), len(result))
		}

		epsilon := float64(1e-4)
		for i, v := range expected {
			if math.Abs(float64(result[0][i]-v)) > epsilon {
				t.Errorf("Element %d mismatch for %s: expected %.6f, got %.6f", i, model, v, result[0][i])
			}
		}
	}
}

func TestMultilingualE5Large(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large model test in short mode")
	}

	fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model: fastembed.MultilingualE5Large,
	})
	if err != nil {
		t.Fatalf("Failed to initialize MultilingualE5Large: %v", err)
	}
	defer fe.Destroy()

	// Test basic embedding
	input := []string{"hello world", "hola mundo", "bonjour le monde"}
	result, err := fe.Embed(input, 1)
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

	// Basic sanity check - embeddings should be normalized
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
	// Test that pooling can be overridden
	meanPooling := fastembed.PoolingMean
	fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:   fastembed.BGESmallENV15,
		Pooling: &meanPooling,
	})
	if err != nil {
		t.Fatalf("Failed to initialize with pooling override: %v", err)
	}
	defer fe.Destroy()

	input := []string{"hello world"}
	result, err := fe.Embed(input, 1)
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

	fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model: fastembed.MultilingualE5LargeInstruct,
	})
	if err != nil {
		t.Fatalf("Failed to initialize E5LargeInstruct: %v", err)
	}
	defer fe.Destroy()

	task := "Given a query, retrieve relevant passages"
	texts := []string{"What is machine learning?", "How do neural networks work?"}

	result, err := fe.InstructEmbed(texts, task, 1)
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
	// Skip if CUDA is not available
	if os.Getenv("TEST_CUDA") != "1" {
		t.Skip("Skipping CUDA test (set TEST_CUDA=1 to enable)")
	}

	fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model:   fastembed.BGESmallENV15,
		UseCUDA: true,
	})
	if err != nil {
		t.Fatalf("Failed to initialize with CUDA: %v", err)
	}
	defer fe.Destroy()

	input := []string{"hello world"}
	result, err := fe.Embed(input, 1)
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

	fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model: fastembed.EmbeddingGemma300M,
	})
	if err != nil {
		t.Fatalf("Failed to initialize EmbeddingGemma300M: %v", err)
	}
	defer fe.Destroy()

	// Test basic embedding
	input := []string{"hello world", "what is machine learning?"}
	result, err := fe.Embed(input, 1)
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

	// Basic sanity check - embeddings should be normalized
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

	t.Logf("EmbeddingGemma embedding successful, first 5 values: %v", result[0][:5])
}

func TestListSupportedModels(t *testing.T) {
	models := fastembed.ListSupportedModels()

	// Check that we have the expected number of models (10 total)
	if len(models) < 10 {
		t.Errorf("Expected at least 10 models, got %d", len(models))
	}

	// Check that new models are included
	modelSet := make(map[fastembed.EmbeddingModel]bool)
	for _, m := range models {
		modelSet[m.Model] = true
	}

	expectedModels := []fastembed.EmbeddingModel{
		fastembed.MultilingualE5Large,
		fastembed.MultilingualE5LargeInstruct,
		fastembed.EmbeddingGemma300M,
		fastembed.EmbeddingGemma300MQ4,
	}

	for _, model := range expectedModels {
		if !modelSet[model] {
			t.Errorf("Expected model %s to be in supported models list", model)
		}
	}
}
