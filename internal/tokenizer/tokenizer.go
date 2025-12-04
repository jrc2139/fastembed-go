// Package tokenizer provides tokenizer loading and management.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Tokenizer wraps the HuggingFace tokenizer with configuration.
type Tokenizer struct {
	*tokenizer.Tokenizer
	maxLength int
}

// MaxLength returns the maximum sequence length.
func (t *Tokenizer) MaxLength() int {
	return t.maxLength
}

// LoadFromPath loads a tokenizer from a model directory.
func LoadFromPath(modelPath string, maxLength int) (*Tokenizer, error) {
	// Load base tokenizer
	tknzer, err := pretrained.FromFile(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer.json: %w", err)
	}

	// Load config files
	config, err := loadJSONFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load config.json: %w", err)
	}

	tokenizerConfig, err := loadJSONFile(filepath.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer_config.json: %w", err)
	}

	tokensMap, err := loadJSONFile(filepath.Join(modelPath, "special_tokens_map.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load special_tokens_map.json: %w", err)
	}

	// Configure tokenizer
	maxLength = configureTokenizer(tknzer, config, tokenizerConfig, tokensMap, maxLength)

	return &Tokenizer{
		Tokenizer: tknzer,
		maxLength: maxLength,
	}, nil
}

// LoadFromBytes loads a tokenizer from raw bytes.
// Note: This requires writing the bytes to a temporary file since the
// underlying tokenizer library only supports loading from files.
func LoadFromBytes(files Files, maxLength int) (*Tokenizer, error) {
	// The sugarme/tokenizer library doesn't support loading from bytes directly.
	// We need to write to temp files and load from there.
	// For now, return an error - this will be implemented when BYOM is fully supported.
	return nil, fmt.Errorf("LoadFromBytes not yet implemented; tokenizer library requires file-based loading")
}

// configureTokenizer applies configuration to a tokenizer.
func configureTokenizer(
	tknzer *tokenizer.Tokenizer,
	config, tokenizerConfig, tokensMap map[string]any,
	maxLength int,
) int {
	// Get model max length from config
	if modelMaxLen, ok := tokenizerConfig["model_max_length"].(float64); ok {
		configMax := int(min(float64(math.MaxInt32), math.Abs(modelMaxLen)))
		if maxLength <= 0 || configMax < maxLength {
			maxLength = configMax
		}
	}

	// Set truncation
	tknzer.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxLength,
		Strategy:  tokenizer.LongestFirst,
		Stride:    0,
	})

	// Set padding
	padID := 0
	if id, ok := config["pad_token_id"].(float64); ok {
		padID = int(id)
	}

	padToken := "[PAD]"
	if token, ok := tokenizerConfig["pad_token"].(string); ok {
		padToken = token
	}

	tknzer.WithPadding(&tokenizer.PaddingParams{
		Strategy:  *tokenizer.NewPaddingStrategy(),
		Direction: tokenizer.Right,
		PadId:     padID,
		PadToken:  padToken,
		PadTypeId: 0,
	})

	// Add special tokens
	specialTokens := parseSpecialTokens(tokensMap)
	if len(specialTokens) > 0 {
		tknzer.AddSpecialTokens(specialTokens)
	}

	return maxLength
}

// parseSpecialTokens extracts special tokens from the tokens map.
func parseSpecialTokens(tokensMap map[string]any) []tokenizer.AddedToken {
	tokens := make([]tokenizer.AddedToken, 0)

	for _, v := range tokensMap {
		switch t := v.(type) {
		case map[string]any:
			token := tokenizer.AddedToken{
				Content: getString(t, "content"),
			}
			if sw, ok := t["single_word"].(bool); ok {
				token.SingleWord = sw
			}
			if ls, ok := t["lstrip"].(bool); ok {
				token.LStrip = ls
			}
			if rs, ok := t["rstrip"].(bool); ok {
				token.RStrip = rs
			}
			if n, ok := t["normalized"].(bool); ok {
				token.Normalized = n
			}
			tokens = append(tokens, token)
		case string:
			tokens = append(tokens, tokenizer.AddedToken{Content: t})
		}
	}

	return tokens
}

// loadJSONFile loads and parses a JSON file.
func loadJSONFile(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseJSON(data)
}

// parseJSON parses JSON bytes into a map.
func parseJSON(data []byte) (map[string]any, error) {
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// getString safely extracts a string from a map.
func getString(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
