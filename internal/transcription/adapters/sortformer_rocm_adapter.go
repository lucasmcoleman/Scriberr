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

// SortformerROCmAdapter runs NVIDIA NeMo Sortformer speaker diarization on AMD
// ROCm (gfx1151). Shares the NeMo venv with the Parakeet ROCm adapter.
// Registered under the "sortformer" model ID on ROCm images.
type SortformerROCmAdapter struct {
	*BaseAdapter
	envPath string
}

const sortformerROCmScript = "sortformer_rocm_diarize.py"

func NewSortformerROCmAdapter(envPath string) *SortformerROCmAdapter {
	capabilities := interfaces.ModelCapabilities{
		ModelID:           "sortformer",
		ModelFamily:       "nvidia_sortformer",
		DisplayName:       "NVIDIA Sortformer (AMD GPU)",
		Description:       "NVIDIA NeMo Sortformer speaker diarization on the AMD ROCm GPU (up to 4 speakers).",
		Version:           "1.0.0",
		SupportedFormats:  []string{"wav", "mp3", "flac", "m4a", "ogg"},
		RequiresGPU:       false,
		MemoryRequirement: 4096,
		Features:          map[string]bool{"diarization": true, "rocm": true},
		Metadata:          map[string]string{"engine": "nvidia_nemo", "framework": "nemo", "backend": "rocm", "max_speakers": "4"},
	}
	schema := []interfaces.ParameterSchema{
		{Name: "model", Type: "string", Required: false, Default: "nvidia/diar_streaming_sortformer_4spk-v2",
			Options: []string{"nvidia/diar_streaming_sortformer_4spk-v2"}, Description: "Sortformer model (streaming, bounded memory)", Group: "basic"},
	}
	base := NewBaseAdapter("sortformer", envPath, capabilities, schema)
	return &SortformerROCmAdapter{BaseAdapter: base, envPath: envPath}
}

func (s *SortformerROCmAdapter) GetMaxSpeakers() int { return 4 }
func (s *SortformerROCmAdapter) GetMinSpeakers() int { return 1 }

func (s *SortformerROCmAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing Sortformer (ROCm) environment", "env_path", s.envPath)
	if err := copyNemoRocmScript(s.envPath, sortformerROCmScript); err != nil {
		return fmt.Errorf("failed to copy sortformer script: %w", err)
	}
	if err := setupNemoRocmEnv(s.envPath); err != nil {
		return fmt.Errorf("failed to setup NeMo ROCm env: %w", err)
	}
	s.initialized = true
	logger.Info("Sortformer (ROCm) environment ready")
	return nil
}

func (s *SortformerROCmAdapter) Diarize(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.DiarizationResult, error) {
	startTime := time.Now()
	s.LogProcessingStart(input, procCtx)
	defer func() { s.LogProcessingEnd(procCtx, time.Since(startTime), nil) }()

	if err := s.ValidateAudioInput(input); err != nil {
		return nil, fmt.Errorf("invalid audio input: %w", err)
	}
	tempDir, err := s.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer s.CleanupTempDirectory(tempDir)

	outputFile := filepath.Join(tempDir, "result.json")
	args := []string{filepath.Join(s.envPath, sortformerROCmScript), input.FilePath, outputFile}
	if model := s.GetStringParameter(params, "model"); model != "" {
		args = append(args, "--model-id", model)
	}

	cmd := exec.CommandContext(ctx, nemoRocmVenvPython(s.envPath), args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	logFile, ferr := os.OpenFile(filepath.Join(procCtx.OutputDirectory, "diarization.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if ferr == nil {
		defer logFile.Close()
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	logger.Info("Executing Sortformer (ROCm) command", "args", strings.Join(args, " "))
	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("diarization was cancelled")
		}
		logTail, _ := s.ReadLogTail(filepath.Join(procCtx.OutputDirectory, "diarization.log"), 2048)
		return nil, fmt.Errorf("Sortformer execution failed: %w\nLogs:\n%s", err, logTail)
	}

	data, err := os.ReadFile(outputFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}
	var raw struct {
		Segments []struct {
			Start      float64 `json:"start"`
			End        float64 `json:"end"`
			Speaker    string  `json:"speaker"`
			Confidence float64 `json:"confidence"`
		} `json:"segments"`
		Speakers    []string `json:"speakers"`
		NumSpeakers int      `json:"num_speakers"`
		Model       string   `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse JSON result: %w", err)
	}
	result := &interfaces.DiarizationResult{
		Segments:       make([]interfaces.DiarizationSegment, len(raw.Segments)),
		SpeakerCount:   raw.NumSpeakers,
		Speakers:       raw.Speakers,
		ProcessingTime: time.Since(startTime),
		ModelUsed:      raw.Model,
	}
	for i, seg := range raw.Segments {
		result.Segments[i] = interfaces.DiarizationSegment{
			Start: seg.Start, End: seg.End, Speaker: seg.Speaker, Confidence: seg.Confidence,
		}
	}
	logger.Info("Sortformer (ROCm) diarization completed", "segments", len(result.Segments), "speakers", result.SpeakerCount)
	return result, nil
}
