package rerank

import (
	"testing"
)

func TestRerank(t *testing.T) {
	reranker, err := New(
		WithModel(BGERerankerBase),
		WithShowDownloadProgress(true),
		WithCUDA(0),
	)
	if err != nil {
		t.Fatalf("failed to create reranker: %v", err)
	}
	defer reranker.Destroy()

	query := "What is a panda?"
	documents := []string{
		"hi there",
		"The giant panda is a bear species endemic to China",
		"A panda is a large black and white mammal",
		"I don't know what you're asking about",
		"Pandas eat bamboo and live in forests",
	}

	results, err := reranker.Rerank(query, documents, true)
	if err != nil {
		t.Fatalf("rerank failed: %v", err)
	}

	if len(results) != len(documents) {
		t.Errorf("expected %d results, got %d", len(documents), len(results))
	}

	// Top results should be panda-related documents
	t.Logf("Query: %s", query)
	t.Logf("Results (sorted by relevance):")
	for i, r := range results {
		t.Logf("  %d. [%.4f] (idx=%d) %s", i+1, r.Score, r.Index, r.Document)
	}

	// Verify sorting - scores should be descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: score[%d]=%.4f > score[%d]=%.4f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// The panda-related documents should rank higher than "hi there" and "I don't know"
	topIndices := make(map[int]bool)
	for i := 0; i < 3; i++ {
		topIndices[results[i].Index] = true
	}
	if !topIndices[1] && !topIndices[2] && !topIndices[4] {
		t.Error("expected panda-related documents to be in top 3 results")
	}
}

func TestRerankBatch(t *testing.T) {
	reranker, err := New(
		WithModel(BGERerankerBase),
		WithCUDA(0),
	)
	if err != nil {
		t.Fatalf("failed to create reranker: %v", err)
	}
	defer reranker.Destroy()

	query := "programming languages"
	documents := make([]string, 20)
	for i := range documents {
		if i%5 == 0 {
			documents[i] = "Python is a popular programming language"
		} else {
			documents[i] = "The weather is nice today"
		}
	}

	results, err := reranker.RerankWithBatchSize(query, documents, false, 8)
	if err != nil {
		t.Fatalf("rerank failed: %v", err)
	}

	if len(results) != 20 {
		t.Errorf("expected 20 results, got %d", len(results))
	}

	// Programming documents should be ranked higher
	for i := 0; i < 4; i++ {
		if results[i].Index%5 != 0 {
			t.Errorf("expected programming document at rank %d, got index %d", i, results[i].Index)
		}
	}
}
