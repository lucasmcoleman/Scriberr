package adapters

import (
	"context"
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

// ParakeetROCmAdapter runs NVIDIA NeMo Parakeet-TDT on AMD ROCm (gfx1151).
// It uses a self-managed venv with the gfx1151 torch wheels (NeMo's uv-sync
// path can't resolve them) and a from_pretrained script with the CUDA-graph
// decoder disabled. Registered under the "parakeet" model ID on ROCm images.
type ParakeetROCmAdapter struct {
	*BaseAdapter
	envPath string
}

const parakeetROCmScript = "parakeet_rocm_transcribe.py"

func NewParakeetROCmAdapter(envPath string) *ParakeetROCmAdapter {
	capabilities := interfaces.ModelCapabilities{
		ModelID:     "parakeet",
		ModelFamily: "nvidia_parakeet",
		DisplayName: "NVIDIA Parakeet-TDT (AMD GPU)",
		Description: "NVIDIA NeMo Parakeet-TDT on the AMD ROCm GPU. Top-tier accuracy + speed with word-level timestamps.",
		Version:     "1.0.0",
		SupportedLanguages: []string{
			"auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk", "cs",
			"bg", "hr", "da", "et", "fi", "el", "hu", "lv", "lt", "mt", "ro", "sk", "sl", "sv",
		},
		SupportedFormats:  []string{"wav", "mp3", "flac", "m4a", "ogg", "opus"},
		RequiresGPU:       false,
		MemoryRequirement: 8192,
		Features: map[string]bool{
			"timestamps": true, "word_level": true, "multilingual": true,
			"high_quality": true, "fast_inference": true, "rocm": true,
		},
		Metadata: map[string]string{"engine": "nvidia_nemo", "framework": "nemo", "backend": "rocm"},
	}
	schema := []interfaces.ParameterSchema{
		{Name: "language", Type: "string", Required: false, Default: "auto",
			Description: "Language ('auto' to detect)", Group: "basic"},
		{Name: "model", Type: "string", Required: false, Default: "nvidia/parakeet-tdt-0.6b-v3",
			Options:     []string{"nvidia/parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v2"},
			Description: "Parakeet model (v3 multilingual, v2 English-only)", Group: "basic"},
	}
	base := NewBaseAdapter("parakeet", envPath, capabilities, schema)
	return &ParakeetROCmAdapter{BaseAdapter: base, envPath: envPath}
}

func (p *ParakeetROCmAdapter) GetSupportedModels() []string {
	return []string{"nvidia/parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v2"}
}

func (p *ParakeetROCmAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing Parakeet (ROCm) environment", "env_path", p.envPath)
	if err := copyNemoRocmScript(p.envPath, parakeetROCmScript); err != nil {
		return fmt.Errorf("failed to copy parakeet script: %w", err)
	}
	if err := setupNemoRocmEnv(p.envPath); err != nil {
		return fmt.Errorf("failed to setup NeMo ROCm env: %w", err)
	}
	p.initialized = true
	logger.Info("Parakeet (ROCm) environment ready")
	return nil
}

func (p *ParakeetROCmAdapter) Transcribe(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.TranscriptResult, error) {
	startTime := time.Now()
	p.LogProcessingStart(input, procCtx)
	defer func() { p.LogProcessingEnd(procCtx, time.Since(startTime), nil) }()

	if err := p.ValidateAudioInput(input); err != nil {
		return nil, fmt.Errorf("invalid audio input: %w", err)
	}
	if err := p.ValidateParameters(params); err != nil {
		return nil, fmt.Errorf("invalid parameters: %w", err)
	}
	tempDir, err := p.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer p.CleanupTempDirectory(tempDir)

	outputFile := filepath.Join(tempDir, "result.json")
	args := []string{filepath.Join(p.envPath, parakeetROCmScript), input.FilePath, outputFile}
	if language := p.GetStringParameter(params, "language"); language != "" {
		args = append(args, "--language", language)
	}
	if model := p.GetStringParameter(params, "model"); model != "" {
		args = append(args, "--model-id", model)
	}

	cmd := exec.CommandContext(ctx, nemoRocmVenvPython(p.envPath), args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	logFile, ferr := os.OpenFile(filepath.Join(procCtx.OutputDirectory, "transcription.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if ferr == nil {
		defer logFile.Close()
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	logger.Info("Executing Parakeet (ROCm) command", "args", strings.Join(args, " "))
	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("transcription was cancelled")
		}
		logTail, _ := p.ReadLogTail(filepath.Join(procCtx.OutputDirectory, "transcription.log"), 2048)
		return nil, fmt.Errorf("Parakeet execution failed: %w\nLogs:\n%s", err, logTail)
	}

	result, err := parseWhisperLikeResult(outputFile)
	if err != nil {
		return nil, fmt.Errorf("failed to parse result: %w", err)
	}
	result.ProcessingTime = time.Since(startTime)
	if result.ModelUsed == "" {
		result.ModelUsed = p.GetStringParameter(params, "model")
	}
	logger.Info("Parakeet (ROCm) transcription completed", "text_length", len(result.Text),
		"segments", len(result.Segments), "words", len(result.WordSegments))
	return result, nil
}

func (p *ParakeetROCmAdapter) GetEstimatedProcessingTime(input interfaces.AudioInput) time.Duration {
	return time.Duration(float64(p.BaseAdapter.GetEstimatedProcessingTime(input)) * 0.15)
}

// parseWhisperLikeResult reads the {text,language,segments,word_segments} JSON
// shape emitted by the whisper_hf and parakeet_rocm scripts.
func parseWhisperLikeResult(resultFile string) (*interfaces.TranscriptResult, error) {
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
		Text: raw.Text, Language: raw.Language, ModelUsed: raw.Model,
		Segments:     make([]interfaces.TranscriptSegment, len(raw.Segments)),
		WordSegments: make([]interfaces.TranscriptWord, len(raw.WordSegments)),
	}
	for i, s := range raw.Segments {
		result.Segments[i] = interfaces.TranscriptSegment{Start: s.Start, End: s.End, Text: s.Text}
	}
	for i, w := range raw.WordSegments {
		result.WordSegments[i] = interfaces.TranscriptWord{Start: w.Start, End: w.End, Word: w.Word, Score: w.Score}
	}
	return result, nil
}
