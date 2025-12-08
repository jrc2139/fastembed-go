package fastembed_test

import (
	"context"
	"sync"
	"testing"

	"github.com/anush008/fastembed-go/text"
)

// TestConcurrentEmbedding tests that concurrent calls to Embed are thread-safe
func TestConcurrentEmbedding(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
		text.WithMaxWorkers(8), // Use multiple workers
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	// Run many concurrent embedding requests
	numGoroutines := 10
	textsPerGoroutine := 50

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Each goroutine embeds different texts
			texts := make([]string, textsPerGoroutine)
			for j := 0; j < textsPerGoroutine; j++ {
				texts[j] = "This is test text number " + string(rune('0'+j%10))
			}

			// Call Embed (which internally uses worker pool)
			embeddings, err := emb.Embed(ctx, texts, 10)
			if err != nil {
				errors <- err
				return
			}

			// Verify embeddings are valid
			if len(embeddings) != textsPerGoroutine {
				t.Errorf("Goroutine %d: expected %d embeddings, got %d", id, textsPerGoroutine, len(embeddings))
			}

			for j, emb := range embeddings {
				if len(emb) != 384 { // BGE Small dimension
					t.Errorf("Goroutine %d, embedding %d: wrong dimension %d", id, j, len(emb))
				}

				// Check for NaN
				for k, v := range emb {
					if v != v { // NaN check
						t.Errorf("Goroutine %d, embedding %d, index %d: NaN value", id, j, k)
					}
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("Concurrent embedding error: %v", err)
	}

	t.Logf("Successfully ran %d concurrent embedding requests with %d texts each",
		numGoroutines, textsPerGoroutine)
}

// TestConcurrentEmbeddingStress is a stress test for concurrent access
func TestConcurrentEmbeddingStress(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
		text.WithMaxWorkers(16),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	// Stress test: many goroutines, many iterations
	numGoroutines := 20
	iterations := 10

	var wg sync.WaitGroup
	var errorCount int
	var mu sync.Mutex

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			for iter := 0; iter < iterations; iter++ {
				texts := []string{
					"Hello world",
					"Machine learning is great",
					"Natural language processing",
				}

				embeddings, err := emb.Embed(ctx, texts, 3)
				if err != nil {
					mu.Lock()
					errorCount++
					mu.Unlock()
					t.Errorf("Goroutine %d, iteration %d: %v", id, iter, err)
					return
				}

				if len(embeddings) != 3 {
					mu.Lock()
					errorCount++
					mu.Unlock()
					t.Errorf("Wrong embedding count")
				}
			}
		}(i)
	}

	wg.Wait()

	if errorCount > 0 {
		t.Errorf("Had %d errors during stress test", errorCount)
	} else {
		t.Logf("Stress test passed: %d goroutines x %d iterations = %d total embedding calls",
			numGoroutines, iterations, numGoroutines*iterations)
	}
}

// TestRaceDetection runs with -race flag to detect data races
func TestRaceDetection(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
		text.WithMaxWorkers(4),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Run concurrent operations that would trigger race detector if unsafe
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			texts := []string{"test one", "test two", "test three"}
			_, _ = emb.Embed(ctx, texts, 3)
		}()
	}
	wg.Wait()

	t.Log("Race detection test passed (run with -race flag for full detection)")
}
