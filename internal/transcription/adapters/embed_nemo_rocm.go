package adapters

import (
	"embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"scriberr/pkg/logger"
)

//go:embed py/nemo_rocm/*
var nemoRocmScripts embed.FS

// nemoRocmTorchIndex is the AMD gfx1151 wheel index (overridable via PYTORCH_INDEX_URL).
const nemoRocmTorchIndex = "https://repo.amd.com/rocm/whl/gfx1151/"

// nemoRocmVenvPython returns the interpreter for a shared NeMo-ROCm env.
func nemoRocmVenvPython(envPath string) string {
	return filepath.Join(envPath, ".venv", "bin", "python")
}

// nemoRocmEnvReady reports whether the env's venv exists and NeMo imports.
func nemoRocmEnvReady(envPath string) bool {
	py := nemoRocmVenvPython(envPath)
	if _, err := os.Stat(py); err != nil {
		return false
	}
	return exec.Command(py, "-c", "import nemo.collections.asr").Run() == nil
}

// copyNemoRocmScript writes one embedded script into the env directory.
func copyNemoRocmScript(envPath, name string) error {
	if err := os.MkdirAll(envPath, 0755); err != nil {
		return err
	}
	data, err := nemoRocmScripts.ReadFile("py/nemo_rocm/" + name)
	if err != nil {
		return fmt.Errorf("read embedded %s: %w", name, err)
	}
	return os.WriteFile(filepath.Join(envPath, name), data, 0755)
}

// setupNemoRocmEnv provisions the shared NeMo venv on AMD ROCm: torch + its
// self-contained ROCm runtime from the gfx1151 index, then nemo-toolkit[asr]
// (pinned) + audio deps from PyPI. Two-phase, like the whisper_hf engine,
// because uv's resolver can't satisfy the gfx1151 torch wheels.
func setupNemoRocmEnv(envPath string) error {
	if nemoRocmEnvReady(envPath) {
		return nil
	}
	if err := os.MkdirAll(envPath, 0755); err != nil {
		return fmt.Errorf("create nemo_rocm dir: %w", err)
	}
	index := GetPyTorchIndexURL(nemoRocmTorchIndex)
	py := nemoRocmVenvPython(envPath)
	run := func(args ...string) error {
		cmd := exec.Command("uv", args...)
		cmd.Dir = envPath
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("uv %s failed: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
		}
		return nil
	}
	logger.Info("Creating NeMo (ROCm) venv", "env_path", envPath)
	if err := run("venv", filepath.Join(envPath, ".venv"), "--python", "3.12"); err != nil {
		return err
	}
	logger.Info("Installing torch + ROCm runtime", "index", index)
	if err := run("pip", "install", "--python", py, "--index-url", index, "torch", "torchaudio", "numpy"); err != nil {
		return err
	}
	logger.Info("Installing nemo-toolkit[asr] + audio deps from PyPI")
	if err := run("pip", "install", "--python", py, "nemo-toolkit[asr]==2.7.3", "librosa", "soundfile"); err != nil {
		return err
	}
	return nil
}
