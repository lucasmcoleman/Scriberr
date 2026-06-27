#!/usr/bin/env python3
"""
NVIDIA Sortformer speaker diarization for Scriberr on AMD ROCm (gfx1151).

Sortformer is a Transformer encoder that emits speaker activity directly (no
RNNT/numba/CUDA-graph decoder), so it runs cleanly on ROCm/HIP. Outputs the
diarization JSON Scriberr's adapter consumes:
  { "segments":[{start,end,speaker}], "speakers":[...], "num_speakers":N }
"""
import argparse
import json
import os
import sys
import time

import torch

if getattr(torch.version, "hip", None):
    try:
        import nemo.core.utils.cuda_python_utils as _cgu
        _cgu.check_cuda_python_cuda_graphs_conditional_nodes_supported = (
            lambda *a, **k: (_ for _ in ()).throw(ImportError("cuda graphs unavailable on ROCm"))
        )
    except Exception:
        pass

from nemo.collections.asr.models import SortformerEncLabelModel


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _parse_entry(e):
    # NeMo sortformer returns "start end speaker_N" strings, or tuples/lists.
    if isinstance(e, str):
        parts = e.split()
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), parts[2]
    elif isinstance(e, (list, tuple)) and len(e) >= 3:
        return float(e[0]), float(e[1]), str(e[2])
    return None


def diarize(audio_path, output_path, model_id):
    if not torch.cuda.is_available():
        _log("FATAL: GPU not available to torch. Refusing to run Sortformer on CPU.")
        sys.exit(2)
    device = "cuda"
    _log(f"device={device} | torch={torch.__version__} hip={getattr(torch.version,'hip',None)} | model={model_id}")

    m = SortformerEncLabelModel.from_pretrained(model_id)
    m = m.to(device).eval()
    _log(f"model on {next(m.parameters()).device}; diarizing {audio_path}")

    t0 = time.time()
    out = m.diarize(audio=[audio_path], batch_size=1)
    # out is a list (per file) of lists of segment entries
    entries = out[0] if out and isinstance(out, list) else []

    segments, speakers = [], []
    for e in entries:
        parsed = _parse_entry(e)
        if not parsed:
            continue
        start, end, spk = parsed
        # Relabel speaker_0 -> Speaker 1 by first appearance
        if spk not in speakers:
            speakers.append(spk)
        label = f"Speaker {speakers.index(spk) + 1}"
        segments.append({"start": round(start, 3), "end": round(end, 3),
                         "speaker": label, "confidence": 1.0})

    result = {
        "segments": segments,
        "speakers": [f"Speaker {i+1}" for i in range(len(speakers))],
        "num_speakers": len(speakers),
        "model": model_id,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"done: {len(segments)} segments, {len(speakers)} speakers in {time.time()-t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description="NeMo Sortformer diarization (ROCm)")
    p.add_argument("audio_path")
    p.add_argument("output_path")
    # The STREAMING v2 model bounds memory on long audio (the offline v1 loads
    # the whole file and OOMs the APU's unified memory on hour-long meetings).
    p.add_argument("--model-id", default=os.environ.get("SORTFORMER_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"))
    args = p.parse_args()
    if not os.path.exists(args.audio_path):
        _log(f"audio not found: {args.audio_path}")
        sys.exit(1)
    try:
        diarize(args.audio_path, args.output_path, args.model_id)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _log(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
