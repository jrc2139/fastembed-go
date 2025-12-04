package tokenizer

// Files contains the raw bytes for tokenizer configuration files.
// Used for loading tokenizers from memory (BYOM support).
type Files struct {
	// TokenizerJSON is the contents of tokenizer.json.
	TokenizerJSON []byte

	// ConfigJSON is the contents of config.json.
	ConfigJSON []byte

	// TokenizerConfigJSON is the contents of tokenizer_config.json.
	TokenizerConfigJSON []byte

	// SpecialTokensMapJSON is the contents of special_tokens_map.json.
	SpecialTokensMapJSON []byte
}

// NewFiles creates a Files struct with required fields.
func NewFiles(tokenizerJSON, configJSON, tokenizerConfigJSON, specialTokensMapJSON []byte) Files {
	return Files{
		TokenizerJSON:        tokenizerJSON,
		ConfigJSON:           configJSON,
		TokenizerConfigJSON:  tokenizerConfigJSON,
		SpecialTokensMapJSON: specialTokensMapJSON,
	}
}
