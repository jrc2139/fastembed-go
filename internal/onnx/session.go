package onnx

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

// Initialize initializes the ONNX Runtime environment.
// Safe to call multiple times - only initializes once.
// Respects ONNX_PATH environment variable for custom library location.
func Initialize() error {
	initOnce.Do(func() {
		// Set library path from environment if provided
		if onnxPath := os.Getenv("ONNX_PATH"); onnxPath != "" {
			ort.SetSharedLibraryPath(onnxPath)
		}

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

// Session wraps an ONNX Runtime dynamic session for efficient inference.
// Uses DynamicAdvancedSession which supports variable input shapes without
// recreating the session for each request.
type Session struct {
	modelPath      string
	options        *ort.SessionOptions
	dynamicSession *ort.DynamicAdvancedSession
	inputNames     []string
	outputNames    []string
}

// NewSession creates a new ONNX session with dynamic input support.
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

// initDynamicSession lazily initializes the dynamic session with specific input/output names.
func (s *Session) initDynamicSession(inputNames, outputNames []string) error {
	if s.dynamicSession != nil {
		return nil
	}

	ds, err := ort.NewDynamicAdvancedSession(
		s.modelPath,
		inputNames,
		outputNames,
		s.options,
	)
	if err != nil {
		return fmt.Errorf("failed to create dynamic session: %w", err)
	}

	s.dynamicSession = ds
	s.inputNames = inputNames
	s.outputNames = outputNames
	return nil
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
	if s.dynamicSession != nil {
		if err := s.dynamicSession.Destroy(); err != nil {
			return err
		}
	}
	if s.options != nil {
		return s.options.Destroy()
	}
	return nil
}

// RunTextEmbedding runs a text embedding inference using a cached dynamic session.
// is2DOutput should be true for models that output direct sentence embeddings (2D: batch, dim),
// false for models that output token embeddings (3D: batch, seq_len, dim).
// Models with custom OutputKey (like "sentence_embedding") typically use 2D output.
func RunTextEmbedding(
	session *Session,
	inputIDs, attentionMask, tokenTypeIDs []int64,
	batchSize, seqLen, dim int,
	outputKey string,
	useTokenTypeIDs bool,
	is2DOutput bool,
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

	// Create output tensor based on expected output shape
	var outputTensor *ort.Tensor[float32]
	if is2DOutput {
		// 2D output: direct sentence embeddings (batch, dim)
		outputShape := ort.NewShape(int64(batchSize), int64(dim))
		outputTensor, err = ort.NewEmptyTensor[float32](outputShape)
	} else {
		// 3D output: token embeddings (batch, seq_len, dim)
		outputShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(dim))
		outputTensor, err = ort.NewEmptyTensor[float32](outputShape)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	outputNames := []string{outputKey}

	// Initialize dynamic session on first use (cached for subsequent calls)
	if err := session.initDynamicSession(inputNames, outputNames); err != nil {
		return nil, nil, err
	}

	// Run inference using cached dynamic session
	if err := session.dynamicSession.Run(inputs, []ort.ArbitraryTensor{outputTensor}); err != nil {
		return nil, nil, fmt.Errorf("failed to run session: %w", err)
	}

	return outputTensor.GetData(), outputTensor.GetShape(), nil
}
