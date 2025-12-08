package fastembed_test

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/anush008/fastembed-go/text"
)

// generateTestTexts creates n test texts of varying lengths
func generateTestTexts(n int) []string {
	templates := []string{
		"Hello world, this is a test.",
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
		"In the realm of natural language processing, embedding models transform text into dense vector representations that capture semantic meaning and enable similarity comparisons.",
		"The development of large language models has revolutionized how we approach tasks like text classification, semantic search, question answering systems, and document summarization.",
	}

	texts := make([]string, n)
	for i := 0; i < n; i++ {
		texts[i] = templates[i%len(templates)]
	}
	return texts
}

// BenchmarkLargeBatch benchmarks embedding with various batch sizes
func BenchmarkLargeBatch(b *testing.B) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		b.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	// Test different total sizes with various batch sizes
	testCases := []struct {
		name      string
		totalSize int
		batchSize int
	}{
		// Small total, varying batch size
		{"total_100_batch_10", 100, 10},
		{"total_100_batch_50", 100, 50},
		{"total_100_batch_100", 100, 100},

		// Medium total, varying batch size
		{"total_500_batch_32", 500, 32},
		{"total_500_batch_64", 500, 64},
		{"total_500_batch_128", 500, 128},
		{"total_500_batch_256", 500, 256},
		{"total_500_batch_500", 500, 500},

		// Large total, varying batch size
		{"total_1000_batch_64", 1000, 64},
		{"total_1000_batch_128", 1000, 128},
		{"total_1000_batch_256", 1000, 256},
		{"total_1000_batch_512", 1000, 512},
		{"total_1000_batch_1000", 1000, 1000},

		// Very large total
		{"total_2000_batch_256", 2000, 256},
		{"total_2000_batch_512", 2000, 512},
		{"total_2000_batch_1000", 2000, 1000},
	}

	for _, tc := range testCases {
		texts := generateTestTexts(tc.totalSize)

		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := emb.Embed(ctx, texts, tc.batchSize)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(tc.totalSize)*float64(b.N)/b.Elapsed().Seconds(), "texts/sec")
		})
	}
}

// BenchmarkBatchAllocation measures allocation overhead at different batch sizes
func BenchmarkBatchAllocation(b *testing.B) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		b.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	batchSizes := []int{32, 64, 128, 256, 512}

	for _, bs := range batchSizes {
		texts := generateTestTexts(bs)

		b.Run("batch_"+string(rune('0'+bs/100))+string(rune('0'+(bs%100)/10))+string(rune('0'+bs%10)), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := emb.Embed(ctx, texts, bs)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestLargeBatchThroughput measures throughput for large batches
func TestLargeBatchThroughput(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	t.Logf("CPU cores: %d", runtime.NumCPU())
	t.Logf("GOMAXPROCS: %d", runtime.GOMAXPROCS(0))

	testCases := []struct {
		total     int
		batchSize int
	}{
		{100, 32},
		{100, 100},
		{500, 64},
		{500, 128},
		{500, 256},
		{500, 500},
		{1000, 128},
		{1000, 256},
		{1000, 512},
		{1000, 1000},
		{2000, 256},
		{2000, 512},
		{2000, 1000},
	}

	for _, tc := range testCases {
		texts := generateTestTexts(tc.total)
		numBatches := (tc.total + tc.batchSize - 1) / tc.batchSize

		// Run multiple iterations
		iterations := 3
		var totalDuration time.Duration

		for i := 0; i < iterations; i++ {
			start := time.Now()
			_, err := emb.Embed(ctx, texts, tc.batchSize)
			if err != nil {
				t.Fatalf("Embed failed: %v", err)
			}
			totalDuration += time.Since(start)
		}

		avgDuration := totalDuration / time.Duration(iterations)
		textsPerSec := float64(tc.total) / avgDuration.Seconds()

		t.Logf("total=%4d batch=%4d batches=%2d: %v avg (%.1f texts/sec)",
			tc.total, tc.batchSize, numBatches, avgDuration.Round(time.Millisecond), textsPerSec)
	}
}

// BenchmarkWorkerPoolScaling tests how worker count affects performance
func BenchmarkWorkerPoolScaling(b *testing.B) {
	ctx := context.Background()
	texts := generateTestTexts(1000)

	workerCounts := []int{1, 2, 4, 8, runtime.NumCPU()}

	for _, workers := range workerCounts {
		b.Run("workers_"+string(rune('0'+workers/10))+string(rune('0'+workers%10)), func(b *testing.B) {
			emb, err := text.New(
				text.WithModel(text.BGESmallENV15),
				text.WithShowDownloadProgress(false),
				text.WithMaxWorkers(workers),
			)
			if err != nil {
				b.Fatalf("Failed to initialize: %v", err)
			}
			defer emb.Destroy()

			// Warmup
			_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := emb.Embed(ctx, texts, 128) // 8 batches
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(1000)*float64(b.N)/b.Elapsed().Seconds(), "texts/sec")
		})
	}
}

// TestOptimalBatchSize verifies the OptimalBatchSize function
func TestOptimalBatchSize(t *testing.T) {
	testCases := []struct {
		name       string
		totalTexts int
		workers    int
		isGPU      bool
		wantMin    int
		wantMax    int
	}{
		{"cpu_small", 100, 4, false, 32, 128},
		{"cpu_medium", 1000, 8, false, 32, 128},
		{"cpu_large", 5000, 16, false, 32, 128},
		{"gpu_small", 100, 4, true, 64, 256},
		{"gpu_medium", 1000, 8, true, 64, 256},
		{"gpu_large", 5000, 16, true, 64, 256},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			bs := text.OptimalBatchSize(tc.totalTexts, tc.workers, tc.isGPU)
			if bs < tc.wantMin || bs > tc.wantMax {
				t.Errorf("OptimalBatchSize(%d, %d, %v) = %d, want in [%d, %d]",
					tc.totalTexts, tc.workers, tc.isGPU, bs, tc.wantMin, tc.wantMax)
			}
			t.Logf("OptimalBatchSize(%d, %d, %v) = %d", tc.totalTexts, tc.workers, tc.isGPU, bs)
		})
	}
}

// TestDefaultBatchSizePerformance compares default batch size (64) vs old default (256)
func TestDefaultBatchSizePerformance(t *testing.T) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()
	texts := generateTestTexts(1000)

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	// Test with default (0 = auto, now 64)
	iterations := 3
	var defaultTime time.Duration
	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := emb.Embed(ctx, texts, 0) // Use default batch size
		if err != nil {
			t.Fatal(err)
		}
		defaultTime += time.Since(start)
	}

	// Test with old default (256)
	var oldDefaultTime time.Duration
	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := emb.Embed(ctx, texts, 256)
		if err != nil {
			t.Fatal(err)
		}
		oldDefaultTime += time.Since(start)
	}

	defaultAvg := defaultTime / time.Duration(iterations)
	oldDefaultAvg := oldDefaultTime / time.Duration(iterations)
	improvement := float64(oldDefaultAvg-defaultAvg) / float64(oldDefaultAvg) * 100

	t.Logf("New default (64): %v avg (%.1f texts/sec)", defaultAvg.Round(time.Millisecond), float64(1000)/defaultAvg.Seconds())
	t.Logf("Old default (256): %v avg (%.1f texts/sec)", oldDefaultAvg.Round(time.Millisecond), float64(1000)/oldDefaultAvg.Seconds())
	t.Logf("Improvement: %.1f%%", improvement)
}

// BenchmarkMemoryPressure tests performance under memory pressure
func BenchmarkMemoryPressure(b *testing.B) {
	emb, err := text.New(
		text.WithModel(text.BGESmallENV15),
		text.WithShowDownloadProgress(false),
	)
	if err != nil {
		b.Fatalf("Failed to initialize: %v", err)
	}
	defer emb.Destroy()

	ctx := context.Background()

	// Warmup
	_, _ = emb.Embed(ctx, []string{"warmup"}, 1)

	// Large single batch (memory intensive)
	b.Run("single_large_batch_1000", func(b *testing.B) {
		texts := generateTestTexts(1000)
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		startAlloc := m.TotalAlloc

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, texts, 1000)
			if err != nil {
				b.Fatal(err)
			}
		}

		runtime.ReadMemStats(&m)
		allocPerOp := (m.TotalAlloc - startAlloc) / uint64(b.N)
		b.ReportMetric(float64(allocPerOp)/1024/1024, "MB/op")
	})

	// Many small batches (allocation overhead)
	b.Run("many_small_batches_1000", func(b *testing.B) {
		texts := generateTestTexts(1000)
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		startAlloc := m.TotalAlloc

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := emb.Embed(ctx, texts, 32) // 32 batches
			if err != nil {
				b.Fatal(err)
			}
		}

		runtime.ReadMemStats(&m)
		allocPerOp := (m.TotalAlloc - startAlloc) / uint64(b.N)
		b.ReportMetric(float64(allocPerOp)/1024/1024, "MB/op")
	})
}
