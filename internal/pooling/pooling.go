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
// Uses a single allocation for all embeddings to reduce memory fragmentation.
func clsPool(data []float32, batchSize, seqLen, dim int) [][]float32 {
	// Single allocation for all embedding data
	allEmbeddings := make([]float32, batchSize*dim)

	// Copy CLS tokens directly into the allocation
	for i := 0; i < batchSize; i++ {
		srcOffset := i * seqLen * dim // Start of this batch item (first token)
		dstOffset := i * dim
		copy(allEmbeddings[dstOffset:dstOffset+dim], data[srcOffset:srcOffset+dim])
	}

	// Create slice headers pointing into the single allocation
	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		result[i] = allEmbeddings[i*dim : (i+1)*dim]
	}
	return result
}

// meanPool computes attention-weighted mean pooling.
// Uses a single allocation for all embeddings to reduce memory fragmentation.
func meanPool(data []float32, batchSize, seqLen, dim int, mask []int64) [][]float32 {
	// Single allocation for all embedding data
	allEmbeddings := make([]float32, batchSize*dim)

	for i := 0; i < batchSize; i++ {
		embedding := allEmbeddings[i*dim : (i+1)*dim]
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
			invMaskSum := 1.0 / maskSum
			for k := 0; k < dim; k++ {
				embedding[k] *= invMaskSum
			}
		}
	}

	// Create slice headers pointing into the single allocation
	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		result[i] = allEmbeddings[i*dim : (i+1)*dim]
	}
	return result
}
