package fastembed_test

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/anush008/fastembed-go/text"
)

// Sample texts of varying lengths for benchmarking
var benchmarkTexts = []string{
	"Hello world",
	"The quick brown fox jumps over the lazy dog.",
	"Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
	"In the realm of natural language processing, embedding models transform text into dense vector representations that capture semantic meaning.",
	"The development of large language models has revolutionized how we approach tasks like text classification, semantic search, and question answering systems.",
}

// Longer texts for throughput testing
var longTexts = []string{
	"Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
	"Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.",
	"Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision and natural language processing.",
	"Natural language processing (NLP) is an interdisciplinary subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The result is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.",
	"Transformers are a type of neural network architecture that has become the foundation for many state-of-the-art natural language processing models. Unlike recurrent neural networks, transformers process all input tokens simultaneously using self-attention mechanisms, allowing them to capture long-range dependencies more effectively and enabling parallel computation during training.",
}

func TestGraniteQ4Basic(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.GraniteEmbeddingEnglishR2Q4),
		text.WithShowDownloadProgress(true),
	)
	if err != nil {
		t.Fatalf("Failed to initialize Granite Q4: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	input := []string{"hello world"}
	result, err := emb.Embed(ctx, input, 1)
	if err != nil {
		t.Fatalf("Failed to embed: %v", err)
	}

	t.Logf("Embedding dimension: %d", len(result[0]))
	t.Logf("First 10 values: %v", result[0][:10])

	// Verify embedding is normalized
	var norm float64
	for _, v := range result[0] {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("Embedding not normalized: norm = %.4f", norm)
	}

	// Check for NaN values
	for i, v := range result[0] {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN value at index %d", i)
		}
	}
}

func BenchmarkGraniteQ4(b *testing.B) {
	emb, err := text.New(
		text.WithModel(text.GraniteEmbeddingEnglishR2Q4),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		b.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	b.Run("single_short", func(b *testing.B) {
		input := []string{"hello world"}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, input, 1)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("single_medium", func(b *testing.B) {
		input := []string{benchmarkTexts[2]}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, input, 1)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("single_long", func(b *testing.B) {
		input := []string{longTexts[0]}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, input, 1)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("batch_5", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, benchmarkTexts, 5)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("batch_5_long", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, longTexts, 5)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func TestGraniteQ4Throughput(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.GraniteEmbeddingEnglishR2Q4),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	// Test different batch sizes
	batchSizes := []int{1, 5, 10, 20, 50}

	for _, batchSize := range batchSizes {
		// Create batch
		batch := make([]string, batchSize)
		for i := 0; i < batchSize; i++ {
			batch[i] = longTexts[i%len(longTexts)]
		}

		// Time the embedding
		iterations := 5
		var totalTime time.Duration

		for i := 0; i < iterations; i++ {
			start := time.Now()
			_, err := emb.Embed(ctx, batch, batchSize)
			if err != nil {
				t.Fatalf("Failed to embed batch of %d: %v", batchSize, err)
			}
			totalTime += time.Since(start)
		}

		avgTime := totalTime / time.Duration(iterations)
		textsPerSecond := float64(batchSize) / avgTime.Seconds()

		t.Logf("Batch size %3d: %v avg, %.1f texts/sec", batchSize, avgTime.Round(time.Millisecond), textsPerSecond)
	}
}

func TestGraniteQ4VsOtherModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping model comparison in short mode")
	}

	models := []text.Model{
		text.GraniteEmbeddingEnglishR2Q4,
		text.BGESmallENV15,
		text.AllMiniLML6V2,
	}

	ctx := context.Background()
	testTexts := benchmarkTexts

	for _, model := range models {
		t.Run(string(model), func(t *testing.T) {
			emb, err := text.New(
				text.WithModel(model),
				text.WithShowDownloadProgress(false),
			)
			if err != nil {
				t.Fatalf("Failed to initialize %s: %v", model, err)
			}
			defer emb.Destroy()

			// Warmup
			_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

			// Benchmark
			iterations := 10
			var totalTime time.Duration

			for i := 0; i < iterations; i++ {
				start := time.Now()
				_, err := emb.Embed(ctx, testTexts, len(testTexts))
				if err != nil {
					t.Fatalf("Failed to embed: %v", err)
				}
				totalTime += time.Since(start)
			}

			avgTime := totalTime / time.Duration(iterations)
			textsPerSecond := float64(len(testTexts)) / avgTime.Seconds()

			t.Logf("Model: %s", model)
			t.Logf("  Dimension: %d", emb.Dimension())
			t.Logf("  Avg time for %d texts: %v", len(testTexts), avgTime.Round(time.Millisecond))
			t.Logf("  Throughput: %.1f texts/sec", textsPerSecond)
		})
	}
}

func TestGraniteSemanticSimilarity(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.GraniteEmbeddingEnglishR2Q4),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Test semantic similarity
	pairs := []struct {
		text1    string
		text2    string
		expected string // "high", "medium", "low"
	}{
		{"The cat sat on the mat", "A feline rested on the rug", "high"},
		{"Machine learning is fascinating", "AI and ML are interesting fields", "high"},
		{"I love pizza", "The weather is nice today", "low"},
		{"Python is a programming language", "JavaScript is used for web development", "medium"},
	}

	texts := make([]string, 0, len(pairs)*2)
	for _, p := range pairs {
		texts = append(texts, p.text1, p.text2)
	}

	embeddings, err := emb.Embed(ctx, texts, len(texts))
	if err != nil {
		t.Fatalf("Failed to embed: %v", err)
	}

	for i, p := range pairs {
		emb1 := embeddings[i*2]
		emb2 := embeddings[i*2+1]

		// Cosine similarity (already normalized)
		var sim float64
		for j := range emb1 {
			sim += float64(emb1[j]) * float64(emb2[j])
		}

		t.Logf("'%s' vs '%s'", p.text1[:min(30, len(p.text1))], p.text2[:min(30, len(p.text2))])
		t.Logf("  Similarity: %.4f (expected: %s)", sim, p.expected)

		// Basic sanity checks
		switch p.expected {
		case "high":
			if sim < 0.7 {
				t.Errorf("Expected high similarity (>0.7), got %.4f", sim)
			}
		case "low":
			if sim > 0.85 {
				t.Errorf("Expected lower similarity (<0.85), got %.4f", sim)
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestGraniteMaxContext(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.GraniteEmbeddingEnglishR2Q4),
		text.WithMaxLength(8192), // Granite supports 8192 tokens
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Generate a long text (roughly 2000 words)
	longText := ""
	sentence := "This is a test sentence that will be repeated many times to create a long document for testing the context window of the Granite embedding model. "
	for len(longText) < 10000 {
		longText += sentence
	}

	t.Logf("Testing with text of %d characters", len(longText))

	start := time.Now()
	result, err := emb.Embed(ctx, []string{longText}, 1)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Failed to embed long text: %v", err)
	}

	t.Logf("Embedding time for long text: %v", elapsed)
	t.Logf("Embedding dimension: %d", len(result[0]))

	// Verify no NaN
	for i, v := range result[0] {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN at index %d", i)
		}
	}

	// Verify normalized
	var norm float64
	for _, v := range result[0] {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("Not normalized: %.4f", norm)
	}

	fmt.Printf("\nGranite Q4 Long Context Test:\n")
	fmt.Printf("  Text length: %d chars\n", len(longText))
	fmt.Printf("  Embedding time: %v\n", elapsed)
	fmt.Printf("  Dimension: %d\n", len(result[0]))
}
