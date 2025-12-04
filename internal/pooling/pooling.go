// Package pooling provides pooling strategies for embedding extraction.
package pooling

// Strategy represents a pooling strategy for extracting embeddings from token outputs.
type Strategy int

const (
	// Cls extracts the CLS token embedding (first token).
	Cls Strategy = iota
	// Mean computes attention-weighted mean of all token embeddings.
	Mean
)

// String returns the string representation of the pooling strategy.
func (s Strategy) String() string {
	switch s {
	case Cls:
		return "cls"
	case Mean:
		return "mean"
	default:
		return "unknown"
	}
}

// Pool applies the pooling strategy to token embeddings.
// data: flattened embeddings [batch * seq * dim]
// batchSize: number of items in batch
// seqLen: sequence length per item
// dim: embedding dimension
// mask: attention mask [batch * seq] (1 = valid token, 0 = padding)
func (s Strategy) Pool(data []float32, batchSize, seqLen, dim int, mask []int64) [][]float32 {
	switch s {
	case Mean:
		return meanPool(data, batchSize, seqLen, dim, mask)
	default:
		return clsPool(data, batchSize, seqLen, dim)
	}
}

// clsPool extracts the CLS token (first token) from each batch item.
func clsPool(data []float32, batchSize, seqLen, dim int) [][]float32 {
	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		embedding := make([]float32, dim)
		offset := i * seqLen * dim // Start of this batch item
		copy(embedding, data[offset:offset+dim])
		result[i] = embedding
	}
	return result
}

// meanPool computes attention-weighted mean pooling.
func meanPool(data []float32, batchSize, seqLen, dim int, mask []int64) [][]float32 {
	result := make([][]float32, batchSize)

	for i := 0; i < batchSize; i++ {
		embedding := make([]float32, dim)
		var maskSum float32

		// Sum embeddings weighted by attention mask
		for j := 0; j < seqLen; j++ {
			maskIdx := i*seqLen + j
			if maskIdx < len(mask) && mask[maskIdx] == 1 {
				maskSum++
				dataOffset := (i*seqLen + j) * dim
				for k := 0; k < dim; k++ {
					embedding[k] += data[dataOffset+k]
				}
			}
		}

		// Divide by mask sum (avoid division by zero)
		if maskSum > 0 {
			for k := 0; k < dim; k++ {
				embedding[k] /= maskSum
			}
		}

		result[i] = embedding
	}

	return result
}
