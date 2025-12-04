// Package download provides model downloading from HuggingFace.
package download

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/schollz/progressbar/v3"
)

// Config holds download configuration.
type Config struct {
	// ModelCode is the HuggingFace repository ID (e.g., "Qdrant/bge-small-en-v1.5-onnx").
	ModelCode string

	// ModelFile is the ONNX model file path within the repo (e.g., "model.onnx").
	ModelFile string

	// AdditionalFiles are extra files to download (e.g., .onnx_data files).
	AdditionalFiles []string

	// TokenizerPath is the subdirectory containing tokenizer files (empty for root).
	TokenizerPath string

	// CacheDir is the local cache directory.
	CacheDir string

	// ShowProgress enables download progress display.
	ShowProgress bool
}

// TokenizerFiles are the standard tokenizer files to download.
var TokenizerFiles = []string{
	"tokenizer.json",
	"config.json",
	"tokenizer_config.json",
	"special_tokens_map.json",
}

// RetrieveModel downloads or retrieves a cached model.
// Returns the path to the model directory.
func RetrieveModel(cfg Config) (string, error) {
	modelDir := filepath.Join(cfg.CacheDir, sanitizePath(cfg.ModelCode))

	// Check if model file already exists
	modelFilePath := filepath.Join(modelDir, filepath.Base(cfg.ModelFile))
	if _, err := os.Stat(modelFilePath); err == nil {
		return modelDir, nil
	}

	return downloadModel(cfg, modelDir)
}

// downloadModel downloads the model from HuggingFace.
func downloadModel(cfg Config, modelDir string) (string, error) {
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return "", fmt.Errorf("failed to create model directory: %w", err)
	}

	// Build list of files to download
	files := buildFileList(cfg)

	if cfg.ShowProgress {
		fmt.Printf("Downloading %s from HuggingFace...\n", cfg.ModelCode)
	}

	endpoint := getHFEndpoint()

	for _, filename := range files {
		url := fmt.Sprintf("%s/%s/resolve/main/%s", endpoint, cfg.ModelCode, filename)
		destPath := filepath.Join(modelDir, filepath.Base(filename))

		if err := downloadFile(url, destPath, cfg.ShowProgress); err != nil {
			return "", fmt.Errorf("failed to download %s: %w", filename, err)
		}
	}

	return modelDir, nil
}

// buildFileList constructs the list of files to download.
func buildFileList(cfg Config) []string {
	files := []string{cfg.ModelFile}

	// Add tokenizer files with optional path prefix
	prefix := cfg.TokenizerPath
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	for _, tf := range TokenizerFiles {
		files = append(files, prefix+tf)
	}

	// Add additional files
	files = append(files, cfg.AdditionalFiles...)

	return files
}

// downloadFile downloads a single file from a URL.
func downloadFile(url, destPath string, showProgress bool) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("HTTP %s", resp.Status)
	}

	destFile, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer destFile.Close()

	var reader io.Reader = resp.Body

	if showProgress {
		bar := progressbar.DefaultBytes(
			resp.ContentLength,
			fmt.Sprintf("Downloading %s", filepath.Base(destPath)),
		)
		pr := progressbar.NewReader(resp.Body, bar)
		reader = &pr
	}

	_, err = io.Copy(destFile, reader)
	return err
}

// getHFEndpoint returns the HuggingFace endpoint URL.
// Respects HF_ENDPOINT environment variable for mirrors.
func getHFEndpoint() string {
	if endpoint := os.Getenv("HF_ENDPOINT"); endpoint != "" {
		return strings.TrimSuffix(endpoint, "/")
	}
	return "https://huggingface.co"
}

// sanitizePath converts a model code to a safe directory name.
func sanitizePath(modelCode string) string {
	// Replace slashes with dashes for nested repos
	return strings.ReplaceAll(modelCode, "/", "--")
}

// GetCacheDir returns the default cache directory.
// Priority: FASTEMBED_CACHE_DIR > HF_HOME/fastembed > ~/.cache/huggingface/fastembed
func GetCacheDir() string {
	if dir := os.Getenv("FASTEMBED_CACHE_DIR"); dir != "" {
		return dir
	}
	if hfHome := os.Getenv("HF_HOME"); hfHome != "" {
		return filepath.Join(hfHome, "fastembed")
	}
	// Default to standard HuggingFace cache location
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return ".fastembed_cache"
	}
	return filepath.Join(homeDir, ".cache", "huggingface", "fastembed")
}
