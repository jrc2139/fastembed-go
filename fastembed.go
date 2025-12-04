package fastembed

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/schollz/progressbar/v3"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

// Pooling represents the pooling strategy for converting token embeddings to sentence embeddings.
type Pooling int

const (
	// PoolingCls extracts the CLS token embedding (first token).
	PoolingCls Pooling = iota
	// PoolingMean computes the mean of all token embeddings weighted by attention mask.
	PoolingMean
)

// Enum-type representing the available embedding models.
type EmbeddingModel string

const (
	AllMiniLML6V2 EmbeddingModel = "fast-all-MiniLM-L6-v2"
	BGEBaseEN     EmbeddingModel = "fast-bge-base-en"
	BGEBaseENV15  EmbeddingModel = "fast-bge-base-en-v1.5"
	BGESmallEN    EmbeddingModel = "fast-bge-small-en"
	BGESmallENV15 EmbeddingModel = "fast-bge-small-en-v1.5"
	BGESmallZH    EmbeddingModel = "fast-bge-small-zh-v1.5"

	// Multilingual E5 models
	MultilingualE5Large         EmbeddingModel = "multilingual-e5-large"
	MultilingualE5LargeInstruct EmbeddingModel = "multilingual-e5-large-instruct"

	// EmbeddingGemma models
	EmbeddingGemma300M   EmbeddingModel = "embeddinggemma-300m"
	EmbeddingGemma300MQ4 EmbeddingModel = "embeddinggemma-300m-q4"
)

// modelRegistryInfo contains full metadata for a model.
type modelRegistryInfo struct {
	Model           EmbeddingModel
	Dim             int
	Description     string
	ModelCode       string   // HuggingFace repository ID
	ModelFile       string   // Path to ONNX file within repo
	AdditionalFiles []string // Additional files to download (e.g., .onnx_data)
	TokenizerPath   string   // Subdirectory containing tokenizer files (empty = root)
	DefaultPooling  Pooling
	OutputKey       *string // Custom ONNX output key (nil = use "last_hidden_state")
	NoTokenTypeIDs  bool    // If true, model doesn't use token_type_ids input
}

// Helper to create string pointer
func strPtr(s string) *string { return &s }

// modelRegistry contains metadata for all supported models.
var modelRegistry = map[EmbeddingModel]modelRegistryInfo{
	AllMiniLML6V2: {
		Model:          AllMiniLML6V2,
		Dim:            384,
		Description:    "Sentence Transformer model, MiniLM-L6-v2",
		ModelCode:      "Qdrant/all-MiniLM-L6-v2-onnx",
		ModelFile:      "model.onnx",
		DefaultPooling: PoolingCls, // Use CLS for backward compatibility with canonical values
	},
	BGEBaseEN: {
		Model:          BGEBaseEN,
		Dim:            768,
		Description:    "Base English model",
		ModelCode:      "Qdrant/fast-bge-base-en",
		ModelFile:      "model_optimized.onnx",
		DefaultPooling: PoolingCls,
	},
	BGEBaseENV15: {
		Model:          BGEBaseENV15,
		Dim:            768,
		Description:    "v1.5 release of the base English model",
		ModelCode:      "Qdrant/bge-base-en-v1.5-onnx-Q",
		ModelFile:      "model_optimized.onnx",
		DefaultPooling: PoolingCls,
	},
	BGESmallEN: {
		Model:          BGESmallEN,
		Dim:            384,
		Description:    "Fast English model",
		ModelCode:      "Qdrant/bge-small-en",
		ModelFile:      "model_optimized.onnx",
		DefaultPooling: PoolingCls,
	},
	BGESmallENV15: {
		Model:          BGESmallENV15,
		Dim:            384,
		Description:    "Fast, default English model",
		ModelCode:      "Qdrant/bge-small-en-v1.5-onnx-Q",
		ModelFile:      "model_optimized.onnx",
		DefaultPooling: PoolingCls,
	},
	BGESmallZH: {
		Model:          BGESmallZH,
		Dim:            512,
		Description:    "Fast Chinese model",
		ModelCode:      "Qdrant/bge-small-zh-v1.5",
		ModelFile:      "model_optimized.onnx",
		DefaultPooling: PoolingCls,
	},
	MultilingualE5Large: {
		Model:           MultilingualE5Large,
		Dim:             1024,
		Description:     "Multilingual E5 Large - 100 language support",
		ModelCode:       "Qdrant/multilingual-e5-large-onnx",
		ModelFile:       "model.onnx",
		AdditionalFiles: []string{"model.onnx_data"},
		DefaultPooling:  PoolingMean,
		NoTokenTypeIDs:  true,
	},
	MultilingualE5LargeInstruct: {
		Model:           MultilingualE5LargeInstruct,
		Dim:             1024,
		Description:     "Multilingual E5 Large Instruct - task-specific embeddings",
		ModelCode:       "intfloat/multilingual-e5-large-instruct",
		ModelFile:       "onnx/model.onnx",
		AdditionalFiles: []string{"onnx/model.onnx_data"},
		TokenizerPath:   "onnx",
		DefaultPooling:  PoolingMean,
		OutputKey:       strPtr("sentence_embedding"),
	},
	EmbeddingGemma300M: {
		Model:           EmbeddingGemma300M,
		Dim:             768,
		Description:     "Google EmbeddingGemma 300M - FP32",
		ModelCode:       "onnx-community/embeddinggemma-300m-ONNX",
		ModelFile:       "onnx/model.onnx",
		AdditionalFiles: []string{"onnx/model.onnx_data"},
		DefaultPooling:  PoolingMean,
		OutputKey:       strPtr("sentence_embedding"),
	},
	EmbeddingGemma300MQ4: {
		Model:           EmbeddingGemma300MQ4,
		Dim:             768,
		Description:     "Google EmbeddingGemma 300M - Q4 quantized",
		ModelCode:       "onnx-community/embeddinggemma-300m-ONNX",
		ModelFile:       "onnx/model_q4.onnx",
		AdditionalFiles: []string{"onnx/model_q4.onnx_data"},
		DefaultPooling:  PoolingMean,
		OutputKey:       strPtr("sentence_embedding"),
	},
}

// Struct to interface with a FastEmbed model.
type FlagEmbedding struct {
	tokenizer      *tokenizer.Tokenizer
	model          EmbeddingModel
	modelInfo      modelRegistryInfo
	maxLength      int
	modelPath      string
	pooling        Pooling
	useCUDA        bool
	cudaDeviceID   int
	sessionOptions *ort.SessionOptions
}

// Options to initialize a FastEmbed model
// Model: The model to use for embedding
// ExecutionProviders: The execution providers to use for onnxruntime
// MaxLength: The maximum length of the input sequence
// CacheDir: The directory to cache the model files
// ShowDownloadProgress: Whether to show the download progress bar
// Pooling: Override the default pooling strategy for the model
// UseCUDA: Enable CUDA execution provider for GPU acceleration
// CUDADeviceID: GPU device ID to use (default 0)
// NOTE:
// We use a pointer for "ShowDownloadProgress" so that we can distinguish between the user
// not setting this flag and the user setting it to false. We want the default value to be true.
// As Go assigns a default(empty) value of "false" to bools, we can't distinguish
// if the user set it to false or not set at all.
// A pointer to bool will be nil if not set explicitly.
type InitOptions struct {
	Model                EmbeddingModel
	ExecutionProviders   []string
	MaxLength            int
	CacheDir             string
	ShowDownloadProgress *bool
	Pooling              *Pooling // Override default pooling (nil = use model default)
	UseCUDA              bool     // Enable CUDA execution provider
	CUDADeviceID         int      // GPU device ID (default 0)
}

// Struct to represent FastEmbed model information.
type ModelInfo struct {
	Model       EmbeddingModel
	Dim         int
	Description string
}

// Function to initialize a FastEmbed model.
func NewFlagEmbedding(options *InitOptions) (*FlagEmbedding, error) {
	if options == nil {
		options = &InitOptions{}
	}

	if options.CacheDir == "" {
		options.CacheDir = "local_cache"
	}

	if options.Model == "" {
		options.Model = BGESmallENV15
	}

	if options.MaxLength == 0 {
		options.MaxLength = 512
	}

	if options.ShowDownloadProgress == nil {
		showDownloadProgress := true
		options.ShowDownloadProgress = &showDownloadProgress
	}

	// Look up model in registry
	modelInfo, ok := modelRegistry[options.Model]
	if !ok {
		return nil, fmt.Errorf("model %s not found in registry", options.Model)
	}

	// Determine pooling strategy
	pooling := modelInfo.DefaultPooling
	if options.Pooling != nil {
		pooling = *options.Pooling
	}

	if onnxPath := os.Getenv("ONNX_PATH"); onnxPath != "" {
		ort.SetSharedLibraryPath(onnxPath)
	}

	if !ort.IsInitialized() {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, err
		}
	}

	// Create session options (needed for CUDA)
	var sessionOptions *ort.SessionOptions
	if options.UseCUDA {
		var err error
		sessionOptions, err = ort.NewSessionOptions()
		if err != nil {
			return nil, fmt.Errorf("failed to create session options: %w", err)
		}

		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			sessionOptions.Destroy()
			return nil, fmt.Errorf("failed to create CUDA provider options: %w", err)
		}

		err = cudaOpts.Update(map[string]string{
			"device_id": fmt.Sprintf("%d", options.CUDADeviceID),
		})
		if err != nil {
			cudaOpts.Destroy()
			sessionOptions.Destroy()
			return nil, fmt.Errorf("failed to update CUDA options: %w", err)
		}

		err = sessionOptions.AppendExecutionProviderCUDA(cudaOpts)
		cudaOpts.Destroy()
		if err != nil {
			sessionOptions.Destroy()
			return nil, fmt.Errorf("failed to append CUDA execution provider: %w", err)
		}
	}

	modelPath, err := retrieveModel(options.Model, options.CacheDir, *options.ShowDownloadProgress)
	if err != nil {
		if sessionOptions != nil {
			sessionOptions.Destroy()
		}
		return nil, err
	}

	tknzer, err := loadTokenizer(modelPath, options.MaxLength)
	if err != nil {
		if sessionOptions != nil {
			sessionOptions.Destroy()
		}
		return nil, err
	}

	return &FlagEmbedding{
		tokenizer:      tknzer,
		model:          options.Model,
		modelInfo:      modelInfo,
		maxLength:      options.MaxLength,
		modelPath:      modelPath,
		pooling:        pooling,
		useCUDA:        options.UseCUDA,
		cudaDeviceID:   options.CUDADeviceID,
		sessionOptions: sessionOptions,
	}, nil
}

// Function to cleanup the internal onnxruntime environment when it is no longer needed.
func (f *FlagEmbedding) Destroy() error {
	if f.sessionOptions != nil {
		if err := f.sessionOptions.Destroy(); err != nil {
			return err
		}
	}
	return ort.DestroyEnvironment()
}

// Private function to embed a batch of input strings.
func (f *FlagEmbedding) onnxEmbed(input []string) ([][]float32, error) {
	inputs := make([]tokenizer.EncodeInput, len(input))
	for index, v := range input {
		sequence := tokenizer.NewInputSequence(v)
		inputs[index] = tokenizer.NewSingleEncodeInput(sequence)
	}

	encodings, err := f.tokenizer.EncodeBatch(inputs, true)
	if err != nil {
		return nil, err
	}

	inputIdsFlat, inputMaskFlat, inputTypeIdsFlat := make([]int64, 0), make([]int64, 0), make([]int64, 0)
	for _, encoding := range encodings {
		inputIds, inputMask, inputTypeIds := encodingToInt32(
			encoding.GetIds(),
			encoding.GetAttentionMask(),
			encoding.GetTypeIds(),
		)
		inputIdsFlat = append(inputIdsFlat, inputIds...)
		inputMaskFlat = append(inputMaskFlat, inputMask...)
		inputTypeIdsFlat = append(inputTypeIdsFlat, inputTypeIds...)
	}

	batchSize := int64(len(inputs))
	seqLen := int64(encodings[0].Len())
	inputShape := ort.NewShape(batchSize, seqLen)

	inputTensorID, err := ort.NewTensor(inputShape, inputIdsFlat)
	if err != nil {
		return nil, err
	}
	defer inputTensorID.Destroy()

	inputTensorMask, err := ort.NewTensor(inputShape, inputMaskFlat)
	if err != nil {
		return nil, err
	}
	defer inputTensorMask.Destroy()

	inputTensorType, err := ort.NewTensor(inputShape, inputTypeIdsFlat)
	if err != nil {
		return nil, err
	}
	defer inputTensorType.Destroy()

	// Determine output key
	outputKey := "last_hidden_state"
	if f.modelInfo.OutputKey != nil {
		outputKey = *f.modelInfo.OutputKey
	}

	// Determine ONNX model path (use base filename since download flattens paths)
	modelFilePath := filepath.Join(f.modelPath, filepath.Base(f.modelInfo.ModelFile))

	// For models with custom output keys (like Gemma with "sentence_embedding"),
	// the output shape might be 2D (batch, dim) instead of 3D (batch, seq, dim)
	var session *ort.AdvancedSession
	var outputTensor *ort.Tensor[float32]

	if f.modelInfo.OutputKey != nil {
		// Models with direct sentence embeddings (2D output)
		outputShape := ort.NewShape(batchSize, int64(f.modelInfo.Dim))
		outputTensor, err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return nil, err
		}
		defer outputTensor.Destroy()

		session, err = ort.NewAdvancedSession(modelFilePath, []string{
			"input_ids", "attention_mask",
		}, []string{
			outputKey,
		}, []ort.ArbitraryTensor{
			inputTensorID, inputTensorMask,
		}, []ort.ArbitraryTensor{outputTensor},
			f.sessionOptions)
	} else if f.modelInfo.NoTokenTypeIDs {
		// Models without token_type_ids input (e.g., E5 models) with 3D output
		outputShape := ort.NewShape(batchSize, seqLen, int64(f.modelInfo.Dim))
		outputTensor, err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return nil, err
		}
		defer outputTensor.Destroy()

		session, err = ort.NewAdvancedSession(modelFilePath, []string{
			"input_ids", "attention_mask",
		}, []string{
			outputKey,
		}, []ort.ArbitraryTensor{
			inputTensorID, inputTensorMask,
		}, []ort.ArbitraryTensor{outputTensor},
			f.sessionOptions)
	} else {
		// Standard models with token embeddings (3D output)
		outputShape := ort.NewShape(batchSize, seqLen, int64(f.modelInfo.Dim))
		outputTensor, err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return nil, err
		}
		defer outputTensor.Destroy()

		session, err = ort.NewAdvancedSession(modelFilePath, []string{
			"input_ids", "attention_mask", "token_type_ids",
		}, []string{
			outputKey,
		}, []ort.ArbitraryTensor{
			inputTensorID, inputTensorMask, inputTensorType,
		}, []ort.ArbitraryTensor{outputTensor},
			f.sessionOptions)
	}
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, err
	}

	outputData := outputTensor.GetData()
	outputDims := outputTensor.GetShape()

	// Handle different output shapes
	if len(outputDims) == 2 {
		// Direct sentence embeddings (e.g., Gemma) - just normalize
		return getEmbeddings2D(outputData, outputDims), nil
	}

	// 3D output - apply pooling
	return f.applyPooling(outputData, outputDims, inputMaskFlat, seqLen), nil
}

// Function to embed a batch of input strings
// The batchSize parameter controls the number of inputs to embed in a single batch
// The batches are processed in parallel
// Returns the first error encountered if any
// Default batch size is 256.
func (f *FlagEmbedding) Embed(input []string, batchSize int) ([]([]float32), error) {
	if batchSize <= 0 {
		batchSize = 256
	}
	embeddings := make([]([]float32), len(input))
	var wg sync.WaitGroup
	errorCh := make(chan error, len(input))
	// var resultsMutex sync.Mutex

	for i := 0; i < len(input); i += batchSize {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			end := i + batchSize
			if end > len(input) {
				end = len(input)
			}
			batchOut, err := f.onnxEmbed(input[i:end])
			if err != nil {
				errorCh <- err
			}
			// resultsMutex.Lock()
			// defer resultsMutex.Unlock()
			// Removed the mutex as the slice positions being accessed are unique for each goroutine and there is no overlap
			copy(embeddings[i:end], batchOut)
		}(i)
	}
	wg.Wait()
	close(errorCh)

	// We can aggregate the errors if we ever need to
	if len(errorCh) > 0 {
		return nil, <-errorCh
	}
	return embeddings, nil
}

// Function to embed a single input string prefixed with "query: "
// Recommended for generating query embeddings for semantic search.
func (f *FlagEmbedding) QueryEmbed(input string) ([]float32, error) {
	query := "query: " + input
	data, err := f.onnxEmbed([]string{query})
	if err != nil {
		return nil, err
	}
	return data[0], nil
}

// Function to embed string prefixed with "passage: ".
func (f *FlagEmbedding) PassageEmbed(input []string, batchSize int) ([][]float32, error) {
	processedInput := make([]string, len(input))
	for i, v := range input {
		processedInput[i] = "passage: " + v
	}
	return f.Embed(processedInput, batchSize)
}

// InstructEmbed embeds text with instruction prefix for E5-Instruct models.
// The task parameter describes the embedding task (e.g., "Given a query, retrieve relevant passages").
// Format: "Instruct: {task}\nQuery: {text}"
// This method is recommended for MultilingualE5LargeInstruct model.
func (f *FlagEmbedding) InstructEmbed(texts []string, task string, batchSize int) ([][]float32, error) {
	prefixed := make([]string, len(texts))
	for i, text := range texts {
		prefixed[i] = fmt.Sprintf("Instruct: %s\nQuery: %s", task, text)
	}
	return f.Embed(prefixed, batchSize)
}

// InstructQueryEmbed embeds a single query with instruction prefix for E5-Instruct models.
func (f *FlagEmbedding) InstructQueryEmbed(query, task string) ([]float32, error) {
	prefixed := fmt.Sprintf("Instruct: %s\nQuery: %s", task, query)
	data, err := f.onnxEmbed([]string{prefixed})
	if err != nil {
		return nil, err
	}
	return data[0], nil
}

// Function to list the supported FastEmbed models.
func ListSupportedModels() []ModelInfo {
	models := make([]ModelInfo, 0, len(modelRegistry))
	for _, info := range modelRegistry {
		models = append(models, ModelInfo{
			Model:       info.Model,
			Dim:         info.Dim,
			Description: info.Description,
		})
	}
	return models
}

func loadTokenizer(modelPath string, maxLength int) (*tokenizer.Tokenizer, error) {
	tknzer, err := pretrained.FromFile(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, err
	}

	configData, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, err
	}

	var config map[string]interface{}
	err = json.Unmarshal(configData, &config)
	if err != nil {
		return nil, err
	}

	tokenizerConfigData, err := os.ReadFile(filepath.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}

	var tokenizerConfig map[string]interface{}
	err = json.Unmarshal(tokenizerConfigData, &tokenizerConfig)
	if err != nil {
		return nil, err
	}

	tokensMapData, err := os.ReadFile(filepath.Join(modelPath, "special_tokens_map.json"))
	if err != nil {
		return nil, err
	}

	var tokensMap map[string]interface{}
	err = json.Unmarshal(tokensMapData, &tokensMap)
	if err != nil {
		return nil, err
	}

	// Handle overflow when coercing to int, major hassle.
	modelMaxLen := int(min(float64(math.MaxInt32), math.Abs(tokenizerConfig["model_max_length"].(float64))))
	maxLength = min(maxLength, modelMaxLen)

	tknzer.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxLength,
		Strategy:  tokenizer.LongestFirst,
		Stride:    0,
	})

	paddingParams := tokenizer.PaddingParams{
		// Strategy defaults to "BatchLongest"
		Strategy:  *tokenizer.NewPaddingStrategy(),
		Direction: tokenizer.Right,
		PadId:     int(config["pad_token_id"].(float64)),
		PadToken:  tokenizerConfig["pad_token"].(string),
		PadTypeId: 0,
	}
	tknzer.WithPadding(&paddingParams)

	specialTokens := make([]tokenizer.AddedToken, 0)

	for _, v := range tokensMap {
		switch t := v.(type) {
		case map[string]interface{}:
			specialToken := tokenizer.AddedToken{
				Content:    t["content"].(string),
				SingleWord: t["single_word"].(bool),
				LStrip:     t["lstrip"].(bool),
				RStrip:     t["rstrip"].(bool),
				Normalized: t["normalized"].(bool),
			}
			specialTokens = append(specialTokens, specialToken)
		case string:
			specialToken := tokenizer.AddedToken{
				Content: t,
			}
			specialTokens = append(specialTokens, specialToken)
		default:
			panic(fmt.Sprintf("unknown type for special_tokens_map.json%T", t))
		}
	}
	tknzer.AddSpecialTokens(specialTokens)

	return tknzer, nil
}

// Private function to get model information from the model name.
func getModelInfo(model EmbeddingModel) (ModelInfo, error) {
	for _, m := range ListSupportedModels() {
		if m.Model == model {
			return m, nil
		}
	}
	return ModelInfo{}, fmt.Errorf("model %s not found", model)
}

// Private function to retrieve the model from the cache or download it
// Returns the path to the model.
func retrieveModel(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
	if _, err := os.Stat(filepath.Join(cacheDir, string(model))); !errors.Is(err, fs.ErrNotExist) {
		return filepath.Join(cacheDir, string(model)), nil
	}
	return downloadFromHuggingFace(model, cacheDir, showDownloadProgress)
}

// Private function to download the model from HuggingFace.
func downloadFromHuggingFace(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
	info, ok := modelRegistry[model]
	if !ok {
		return "", fmt.Errorf("model %s not found in registry", model)
	}

	modelDir := filepath.Join(cacheDir, string(model))
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return "", err
	}

	// Determine tokenizer path prefix (some models have tokenizer in subdirectory)
	tokenizerPrefix := info.TokenizerPath
	if tokenizerPrefix != "" {
		tokenizerPrefix += "/"
	}

	// List of files to download from HuggingFace
	files := []string{
		info.ModelFile,
		tokenizerPrefix + "tokenizer.json",
		tokenizerPrefix + "config.json",
		tokenizerPrefix + "tokenizer_config.json",
		tokenizerPrefix + "special_tokens_map.json",
	}

	// Add additional files (like .onnx_data)
	files = append(files, info.AdditionalFiles...)

	if showDownloadProgress {
		fmt.Printf("Downloading %s from HuggingFace...\n", model)
	}

	// Download each file
	for _, filename := range files {
		downloadURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", info.ModelCode, filename)

		response, err := http.Get(downloadURL)
		if err != nil {
			return "", fmt.Errorf("failed to download %s: %w", filename, err)
		}

		if response.StatusCode < 200 || response.StatusCode > 299 {
			response.Body.Close()
			return "", fmt.Errorf("failed to download %s: %s", filename, response.Status)
		}

		// Flatten all files to root of model dir (use base filename only)
		destFilename := filepath.Base(filename)
		destPath := filepath.Join(modelDir, destFilename)

		destFile, err := os.Create(destPath)
		if err != nil {
			response.Body.Close()
			return "", fmt.Errorf("failed to create %s: %w", filename, err)
		}

		if showDownloadProgress {
			bar := progressbar.DefaultBytes(
				response.ContentLength,
				fmt.Sprintf("Downloading %s", filepath.Base(filename)),
			)
			reader := progressbar.NewReader(response.Body, bar)
			_, err = io.Copy(destFile, &reader)
		} else {
			_, err = io.Copy(destFile, response.Body)
		}

		destFile.Close()
		response.Body.Close()

		if err != nil {
			return "", fmt.Errorf("failed to write %s: %w", filename, err)
		}
	}

	return modelDir, nil
}

// Private function to untar the downloaded model from a .tar.gz file.
func untar(tarball io.Reader, target string) error {
	archive, err := gzip.NewReader(tarball)
	if err != nil {
		return err
	}
	defer archive.Close()

	tarReader := tar.NewReader(archive)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		path := filepath.Join(target, header.Name)
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(path, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return err
			}

			file, err := os.Create(path)
			if err != nil {
				return err
			}
			if _, err := io.Copy(file, tarReader); err != nil {
				file.Close()
				return err
			}

			file.Close()
		}
	}
	return nil
}

// Private function to normalize a vector
// Based on https://github.com/qdrant/fastembed/blob/ca6f9d629ad14da1dfd094c846976b0c964b32cf/fastembed/embedding.py#L16
func normalize(v []float32) []float32 {
	norm := float32(0.0)
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	epsilon := float32(1e-12)

	normalized := make([]float32, len(v))
	for i, val := range v {
		normalized[i] = (val / norm) + epsilon
	}

	return normalized
}

// Private function to return the normalized embeddings from a flattened array with the given dimensions.
// Used for 3D outputs with CLS pooling (extracts first token).
func getEmbeddings(data []float32, dimensions []int64) [][]float32 {
	x, y, z := dimensions[0], dimensions[1], dimensions[2]
	embeddings := make([][]float32, x)
	var i int64
	for i = 0; i < x; i++ {
		startIndex := i * y * z
		endIndex := startIndex + z
		embeddings[i] = normalize(data[startIndex:endIndex])
	}
	return embeddings
}

// getEmbeddings2D handles 2D output (batch, dim) - used for models that output sentence embeddings directly.
func getEmbeddings2D(data []float32, dimensions []int64) [][]float32 {
	batchSize, dim := dimensions[0], dimensions[1]
	embeddings := make([][]float32, batchSize)
	var i int64
	for i = 0; i < batchSize; i++ {
		startIndex := i * dim
		endIndex := startIndex + dim
		embeddings[i] = normalize(data[startIndex:endIndex])
	}
	return embeddings
}

// applyPooling applies the configured pooling strategy to 3D token embeddings.
func (f *FlagEmbedding) applyPooling(data []float32, dimensions, attentionMask []int64, seqLen int64) [][]float32 {
	switch f.pooling {
	case PoolingMean:
		// Mean pooling: weighted average by attention mask
		return meanPool(data, dimensions, attentionMask, seqLen)
	case PoolingCls:
		fallthrough
	default:
		// CLS pooling: extract first token for each batch item
		return getEmbeddings(data, dimensions)
	}
}

// meanPool computes attention-weighted mean of token embeddings.
func meanPool(data []float32, dimensions, attentionMask []int64, seqLen int64) [][]float32 {
	batchSize := dimensions[0]
	dim := dimensions[2]

	embeddings := make([][]float32, batchSize)

	for b := int64(0); b < batchSize; b++ {
		// Sum embeddings weighted by attention mask
		sum := make([]float32, dim)
		maskSum := float32(0)

		for t := int64(0); t < seqLen; t++ {
			maskIdx := b*seqLen + t
			mask := float32(attentionMask[maskIdx])
			maskSum += mask

			if mask > 0 {
				for d := int64(0); d < dim; d++ {
					dataIdx := b*seqLen*dim + t*dim + d
					sum[d] += data[dataIdx] * mask
				}
			}
		}

		// Avoid division by zero
		if maskSum == 0 {
			maskSum = 1
		}

		// Compute mean
		embedding := make([]float32, dim)
		for d := int64(0); d < dim; d++ {
			embedding[d] = sum[d] / maskSum
		}

		embeddings[b] = normalize(embedding)
	}

	return embeddings
}

// Private function to convert multiple int32 slices to int64 slices as required by the onnxruntime API
// With a linear time complexity.
func encodingToInt32(inputA, inputB, inputC []int) ([]int64, []int64, []int64) {
	if len(inputA) != len(inputB) || len(inputB) != len(inputC) {
		panic("input lengths do not match")
	}
	outputA := make([]int64, len(inputA))
	outputB := make([]int64, len(inputB))
	outputC := make([]int64, len(inputC))
	for i := range inputA {
		outputA[i] = int64(inputA[i])
		outputB[i] = int64(inputB[i])
		outputC[i] = int64(inputC[i])
	}
	return outputA, outputB, outputC
}
