package adapters

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"scriberr/internal/transcription/interfaces"
	"scriberr/pkg/logger"
)

//go:embed py/whisper_hf/*
var whisperHFScripts embed.FS

// defaultWhisperHFIndex is the AMD ROCm gfx1151 wheel index that ships native
// gfx1151 kernels (torch 2.x+rocmX). It is overridable via PYTORCH_INDEX_URL so
// the same engine can target CUDA/CPU on other hosts.
const defaultWhisperHFIndex = "https://repo.amd.com/rocm/whl/gfx1151/"

// WhisperHFAdapter implements TranscriptionAdapter using Hugging Face
// Transformers Whisper (pure PyTorch, SDPA). This is the engine that actually
// runs transcription on AMD ROCm GPUs (gfx1151 / Strix Halo) — it has no
// CTranslate2 or torchvision dependency, so it loads cleanly under HIP.
type WhisperHFAdapter struct {
	*BaseAdapter
	envPath string
}

// NewWhisperHFAdapter creates a new HF Transformers Whisper adapter.
func NewWhisperHFAdapter(envPath string) *WhisperHFAdapter {
	capabilities := interfaces.ModelCapabilities{
		ModelID:     "whisper_hf",
		ModelFamily: "hf_whisper",
		DisplayName: "Whisper (Transformers, AMD/CPU)",
		Description: "OpenAI Whisper via Hugging Face Transformers (SDPA). Pure-PyTorch — runs on AMD ROCm (gfx1151) GPUs and CPU.",
		Version:     "1.0.0",
		SupportedLanguages: []string{
			"auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
			"ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el",
			"ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt",
			"la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl",
			"kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq",
			"sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka",
			"be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk",
			"nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
			"ha", "ba", "jw", "su",
		},
		SupportedFormats:  []string{"wav", "mp3", "flac", "m4a", "ogg", "opus", "webm"},
		RequiresGPU:       false,
		MemoryRequirement: 12288, // large-v3 fp16 peaks ~10-12GB on GPU
		Features: map[string]bool{
			"timestamps":         true,
			"word_level":         true,
			"multilingual":       true,
			"high_quality":       true,
			"fast_inference":     true,
			"transformers_based": true,
			"rocm":               true,
		},
		Metadata: map[string]string{
			"engine":    "openai_whisper",
			"framework": "transformers",
			"attention": "sdpa",
			"backend":   "rocm/cuda/cpu",
			"model_id":  "openai/whisper-large-v3-turbo",
		},
	}

	schema := []interfaces.ParameterSchema{
		{
			Name:        "language",
			Type:        "string",
			Required:    false,
			Default:     "auto",
			Description: "Language of the audio ('auto' to detect)",
			Group:       "basic",
		},
		{
			Name:        "model",
			Type:        "string",
			Required:    false,
			Default:     "openai/whisper-large-v3-turbo",
			Options:     []string{"openai/whisper-large-v3-turbo", "openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small"},
			Description: "Whisper model variant (turbo is fastest; large-v3 most accurate)",
			Group:       "basic",
		},
		{
			Name:        "batch_size",
			Type:        "int",
			Required:    false,
			Default:     8,
			Min:         &[]float64{1}[0],
			Max:         &[]float64{64}[0],
			Description: "Chunk batch size (higher = faster on GPU, more VRAM)",
			Group:       "advanced",
		},
		{
			Name:        "chunk_length",
			Type:        "int",
			Required:    false,
			Default:     30,
			Min:         &[]float64{5}[0],
			Max:         &[]float64{30}[0],
			Description: "Chunk length in seconds (Whisper context is 30s)",
			Group:       "advanced",
		},
	}

	baseAdapter := NewBaseAdapter("whisper_hf", envPath, capabilities, schema)

	return &WhisperHFAdapter{
		BaseAdapter: baseAdapter,
		envPath:     envPath,
	}
}

// GetSupportedModels returns the available Whisper model variants.
func (w *WhisperHFAdapter) GetSupportedModels() []string {
	return []string{
		"openai/whisper-large-v3-turbo",
		"openai/whisper-large-v3",
		"openai/whisper-medium",
		"openai/whisper-small",
	}
}

// PrepareEnvironment sets up the whisper_hf Python environment.
func (w *WhisperHFAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing whisper_hf environment", "env_path", w.envPath)

	if err := w.copyTranscriptionScript(); err != nil {
		return fmt.Errorf("failed to copy transcription script: %w", err)
	}

	// Already provisioned?
	if w.isEnvReady() {
		logger.Info("whisper_hf environment already ready")
		w.initialized = true
		return nil
	}

	if err := w.setupEnvironment(); err != nil {
		return fmt.Errorf("failed to setup whisper_hf environment: %w", err)
	}

	w.initialized = true
	logger.Info("whisper_hf environment prepared successfully")
	return nil
}

// venvPython returns the path to the engine venv's interpreter.
func (w *WhisperHFAdapter) venvPython() string {
	return filepath.Join(w.envPath, ".venv", "bin", "python")
}

// isEnvReady reports whether the venv exists and torch+transformers import.
func (w *WhisperHFAdapter) isEnvReady() bool {
	py := w.venvPython()
	if _, err := os.Stat(py); err != nil {
		return false
	}
	return exec.Command(py, "-c", "import torch, transformers").Run() == nil
}

// runUV runs a uv subcommand in the env dir, surfacing combined output on error.
func (w *WhisperHFAdapter) runUV(args ...string) error {
	cmd := exec.Command("uv", args...)
	cmd.Dir = w.envPath
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("uv %s failed: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return nil
}

// setupEnvironment provisions the venv with a deliberate two-phase install:
// torch + its self-contained ROCm runtime come ONLY from the wheel index
// (default: AMD gfx1151), then the transformers stack comes from PyPI. This
// avoids uv's multi-index resolver, which fails because the AMD index returns
// HTTP 403 (not 404) for packages it does not host.
func (w *WhisperHFAdapter) setupEnvironment() error {
	if err := os.MkdirAll(w.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create whisper_hf directory: %w", err)
	}

	indexURL := GetPyTorchIndexURL(defaultWhisperHFIndex)
	py := w.venvPython()

	logger.Info("Creating whisper_hf venv")
	if err := w.runUV("venv", filepath.Join(w.envPath, ".venv"), "--python", "3.12"); err != nil {
		return err
	}

	logger.Info("Installing torch + ROCm runtime", "index", indexURL)
	if err := w.runUV("pip", "install", "--python", py, "--index-url", indexURL, "torch", "torchaudio", "numpy"); err != nil {
		return err
	}

	logger.Info("Installing transformers stack from PyPI")
	if err := w.runUV("pip", "install", "--python", py, "transformers>=4.46", "accelerate>=0.34", "soundfile"); err != nil {
		return err
	}

	return nil
}

// copyTranscriptionScript writes the embedded Python script into the env dir.
func (w *WhisperHFAdapter) copyTranscriptionScript() error {
	if err := os.MkdirAll(w.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	scriptContent, err := whisperHFScripts.ReadFile("py/whisper_hf/whisper_transcribe.py")
	if err != nil {
		return fmt.Errorf("failed to read embedded whisper_transcribe.py: %w", err)
	}

	scriptPath := filepath.Join(w.envPath, "whisper_transcribe.py")
	if err := os.WriteFile(scriptPath, scriptContent, 0755); err != nil {
		return fmt.Errorf("failed to write transcription script: %w", err)
	}

	return nil
}

// Transcribe runs the HF Whisper engine on the input audio.
func (w *WhisperHFAdapter) Transcribe(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.TranscriptResult, error) {
	startTime := time.Now()
	w.LogProcessingStart(input, procCtx)
	defer func() {
		w.LogProcessingEnd(procCtx, time.Since(startTime), nil)
	}()

	if err := w.ValidateAudioInput(input); err != nil {
		return nil, fmt.Errorf("invalid audio input: %w", err)
	}
	if err := w.ValidateParameters(params); err != nil {
		return nil, fmt.Errorf("invalid parameters: %w", err)
	}

	tempDir, err := w.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer w.CleanupTempDirectory(tempDir)

	args := w.buildArgs(input, params, tempDir)

	cmd := exec.CommandContext(ctx, w.venvPython(), args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	logFile, err := os.OpenFile(filepath.Join(procCtx.OutputDirectory, "transcription.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logger.Warn("Failed to create log file", "error", err)
	} else {
		defer logFile.Close()
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}

	logger.Info("Executing whisper_hf command", "args", strings.Join(args, " "))

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("transcription was cancelled")
		}
		logPath := filepath.Join(procCtx.OutputDirectory, "transcription.log")
		logTail, _ := w.ReadLogTail(logPath, 2048)
		logger.Error("whisper_hf execution failed", "error", err)
		return nil, fmt.Errorf("whisper_hf execution failed: %w\nLogs:\n%s", err, logTail)
	}

	result, err := w.parseResult(tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to parse result: %w", err)
	}

	result.ProcessingTime = time.Since(startTime)
	if result.ModelUsed == "" {
		result.ModelUsed = w.GetStringParameter(params, "model")
	}

	logger.Info("whisper_hf transcription completed",
		"text_length", len(result.Text),
		"segments", len(result.Segments),
		"words", len(result.WordSegments),
		"processing_time", result.ProcessingTime)

	return result, nil
}

// buildArgs assembles the `python whisper_transcribe.py ...` invocation.
func (w *WhisperHFAdapter) buildArgs(input interfaces.AudioInput, params map[string]interface{}, tempDir string) []string {
	outputFile := filepath.Join(tempDir, "result.json")
	scriptPath := filepath.Join(w.envPath, "whisper_transcribe.py")

	args := []string{
		scriptPath,
		input.FilePath,
		outputFile,
	}

	if language := w.GetStringParameter(params, "language"); language != "" {
		args = append(args, "--language", language)
	}
	if model := w.GetStringParameter(params, "model"); model != "" {
		args = append(args, "--model-id", model)
	}
	if bs := w.GetIntParameter(params, "batch_size"); bs > 0 {
		args = append(args, "--batch-size", fmt.Sprintf("%d", bs))
	}
	if cl := w.GetIntParameter(params, "chunk_length"); cl > 0 {
		args = append(args, "--chunk-length", fmt.Sprintf("%d", cl))
	}

	return args
}

// parseResult reads the engine's JSON output into the standard result format.
func (w *WhisperHFAdapter) parseResult(tempDir string) (*interfaces.TranscriptResult, error) {
	resultFile := filepath.Join(tempDir, "result.json")

	data, err := os.ReadFile(resultFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var raw struct {
		Text     string `json:"text"`
		Language string `json:"language"`
		Model    string `json:"model"`
		Segments []struct {
			Start float64 `json:"start"`
			End   float64 `json:"end"`
			Text  string  `json:"text"`
		} `json:"segments"`
		WordSegments []struct {
			Start float64 `json:"start"`
			End   float64 `json:"end"`
			Word  string  `json:"word"`
			Score float64 `json:"score"`
		} `json:"word_segments"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse JSON result: %w", err)
	}

	result := &interfaces.TranscriptResult{
		Text:         raw.Text,
		Language:     raw.Language,
		ModelUsed:    raw.Model,
		Segments:     make([]interfaces.TranscriptSegment, len(raw.Segments)),
		WordSegments: make([]interfaces.TranscriptWord, len(raw.WordSegments)),
	}
	for i, seg := range raw.Segments {
		result.Segments[i] = interfaces.TranscriptSegment{
			Start: seg.Start,
			End:   seg.End,
			Text:  seg.Text,
		}
	}
	for i, wd := range raw.WordSegments {
		result.WordSegments[i] = interfaces.TranscriptWord{
			Start: wd.Start,
			End:   wd.End,
			Word:  wd.Word,
			Score: wd.Score,
		}
	}

	return result, nil
}

// GetEstimatedProcessingTime — GPU Whisper runs well under realtime.
func (w *WhisperHFAdapter) GetEstimatedProcessingTime(input interfaces.AudioInput) time.Duration {
	baseTime := w.BaseAdapter.GetEstimatedProcessingTime(input)
	return time.Duration(float64(baseTime) * 0.2)
}
