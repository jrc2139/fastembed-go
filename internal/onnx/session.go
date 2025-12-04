package onnx

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

// Initialize initializes the ONNX Runtime environment.
// Safe to call multiple times - only initializes once.
func Initialize() error {
	initOnce.Do(func() {
		if !ort.IsInitialized() {
			initErr = ort.InitializeEnvironment()
		}
	})
	return initErr
}

// SessionConfig holds configuration for creating a session.
type SessionConfig struct {
	// ModelPath is the path to the ONNX model file.
	ModelPath string

	// InputNames are the model input names.
	InputNames []string

	// OutputNames are the model output names.
	OutputNames []string

	// Providers are the execution providers to use (optional).
	Providers []ExecutionProvider
}

// Session wraps an ONNX Runtime advanced session.
type Session struct {
	modelPath string
	options   *ort.SessionOptions
}

// NewSession creates a new ONNX session configuration.
func NewSession(modelPath string, providers ...ExecutionProvider) (*Session, error) {
	if err := Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}

	// Apply execution providers
	for _, p := range providers {
		if err := p.Apply(opts); err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("failed to configure %s provider: %w", p.Name(), err)
		}
	}

	return &Session{
		modelPath: modelPath,
		options:   opts,
	}, nil
}

// Options returns the session options for use with AdvancedSession.
func (s *Session) Options() *ort.SessionOptions {
	return s.options
}

// ModelPath returns the model file path.
func (s *Session) ModelPath() string {
	return s.modelPath
}

// Destroy cleans up session resources.
func (s *Session) Destroy() error {
	if s.options != nil {
		return s.options.Destroy()
	}
	return nil
}

// RunTextEmbedding runs a text embedding inference.
// Handles both 2D (sentence embeddings) and 3D (token embeddings) outputs.
func RunTextEmbedding(
	session *Session,
	inputIDs, attentionMask, tokenTypeIDs []int64,
	batchSize, seqLen, dim int,
	outputKey string,
	useTokenTypeIDs bool,
) ([]float32, []int64, error) {
	inputShape := ort.NewShape(int64(batchSize), int64(seqLen))

	// Create input tensors
	inputIDTensor, err := ort.NewTensor(inputShape, inputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	defer inputIDTensor.Destroy()

	maskTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	// Build inputs based on whether model uses token_type_ids
	var inputNames []string
	var inputs []ort.ArbitraryTensor

	if useTokenTypeIDs {
		typeTensor, err := ort.NewTensor(inputShape, tokenTypeIDs)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create token_type_ids tensor: %w", err)
		}
		defer typeTensor.Destroy()

		inputNames = []string{"input_ids", "attention_mask", "token_type_ids"}
		inputs = []ort.ArbitraryTensor{inputIDTensor, maskTensor, typeTensor}
	} else {
		inputNames = []string{"input_ids", "attention_mask"}
		inputs = []ort.ArbitraryTensor{inputIDTensor, maskTensor}
	}

	// Try 2D output first (sentence embeddings), fall back to 3D (token embeddings)
	outputShape2D := ort.NewShape(int64(batchSize), int64(dim))
	outputTensor2D, err := ort.NewEmptyTensor[float32](outputShape2D)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Try 2D session
	advSession, err := ort.NewAdvancedSession(
		session.modelPath,
		inputNames,
		[]string{outputKey},
		inputs,
		[]ort.ArbitraryTensor{outputTensor2D},
		session.options,
	)

	if err == nil {
		defer advSession.Destroy()
		defer outputTensor2D.Destroy()

		if err := advSession.Run(); err != nil {
			return nil, nil, fmt.Errorf("failed to run session: %w", err)
		}

		return outputTensor2D.GetData(), outputTensor2D.GetShape(), nil
	}

	// 2D failed, try 3D output
	outputTensor2D.Destroy()

	outputShape3D := ort.NewShape(int64(batchSize), int64(seqLen), int64(dim))
	outputTensor3D, err := ort.NewEmptyTensor[float32](outputShape3D)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create 3D output tensor: %w", err)
	}
	defer outputTensor3D.Destroy()

	advSession, err = ort.NewAdvancedSession(
		session.modelPath,
		inputNames,
		[]string{outputKey},
		inputs,
		[]ort.ArbitraryTensor{outputTensor3D},
		session.options,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create session: %w", err)
	}
	defer advSession.Destroy()

	if err := advSession.Run(); err != nil {
		return nil, nil, fmt.Errorf("failed to run session: %w", err)
	}

	return outputTensor3D.GetData(), outputTensor3D.GetShape(), nil
}
