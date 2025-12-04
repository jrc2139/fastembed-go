// Package output provides output key handling for ONNX model outputs.
package output

import "fmt"

// KeyType specifies how to select an output from a model.
type KeyType int

const (
	// OnlyOne selects the sole output if exactly one exists.
	OnlyOne KeyType = iota
	// ByOrder selects output by index.
	ByOrder
	// ByName selects output by name.
	ByName
)

// Key specifies which output to select from a model.
type Key struct {
	Type  KeyType
	Index int    // Used when Type is ByOrder
	Name  string // Used when Type is ByName
}

// NewKeyByName creates a Key that selects output by name.
func NewKeyByName(name string) *Key {
	return &Key{Type: ByName, Name: name}
}

// NewKeyByOrder creates a Key that selects output by index.
func NewKeyByOrder(index int) *Key {
	return &Key{Type: ByOrder, Index: index}
}

// NewKeyOnlyOne creates a Key that selects the sole output.
func NewKeyOnlyOne() *Key {
	return &Key{Type: OnlyOne}
}

// String returns a string representation of the key.
func (k *Key) String() string {
	if k == nil {
		return "nil"
	}
	switch k.Type {
	case OnlyOne:
		return "only_one"
	case ByOrder:
		return fmt.Sprintf("by_order(%d)", k.Index)
	case ByName:
		return fmt.Sprintf("by_name(%s)", k.Name)
	default:
		return "unknown"
	}
}

// DefaultTextPrecedence is the default output key precedence for text embedding models.
// It tries these keys in order until one matches.
var DefaultTextPrecedence = []*Key{
	{Type: OnlyOne},
	{Type: ByName, Name: "sentence_embedding"},
	{Type: ByName, Name: "text_embeds"},
	{Type: ByName, Name: "last_hidden_state"},
}

// DefaultSparsePrecedence is the default output key precedence for sparse embedding models.
var DefaultSparsePrecedence = []*Key{
	{Type: OnlyOne},
	{Type: ByName, Name: "output"},
}

// DefaultImagePrecedence is the default output key precedence for image embedding models.
var DefaultImagePrecedence = []*Key{
	{Type: OnlyOne},
	{Type: ByName, Name: "image_embeds"},
}

// DefaultRerankPrecedence is the default output key precedence for reranking models.
var DefaultRerankPrecedence = []*Key{
	{Type: OnlyOne},
	{Type: ByName, Name: "logits"},
}
