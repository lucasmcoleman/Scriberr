#!/usr/bin/env python3
"""
NVIDIA Parakeet-TDT transcription for Scriberr on AMD ROCm (gfx1151).

Runs NeMo Parakeet purely for inference on the AMD GPU. NeMo's CUDA-graph RNNT/TDT
decoder is disabled (it dlopens libcuda.so.1, absent on AMD); the pure-PyTorch
label-looping decoder runs on ROCm via the torch.cuda (HIP) namespace.

Outputs the WhisperX-compatible JSON shape Scriberr consumes:
  { "text", "language", "segments":[{start,end,text}], "word_segments":[{start,end,word,score}] }
"""
import argparse
import json
import os
import sys
import time

import torch

# Disable NeMo's CUDA-graph decoder support check BEFORE importing the ASR
# collection (it raises on AMD by dlopen'ing libcuda). No-op on NVIDIA CUDA.
if getattr(torch.version, "hip", None):
    try:
        import nemo.core.utils.cuda_python_utils as _cgu
        _cgu.check_cuda_python_cuda_graphs_conditional_nodes_supported = (
            lambda *a, **k: (_ for _ in ()).throw(ImportError("cuda graphs unavailable on ROCm"))
        )
    except Exception:
        pass

import nemo.collections.asr as nemo_asr


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _seconds(entry, key, stride_keys):
    """Extract a timestamp in seconds from a NeMo timestamp entry."""
    if key in entry and entry[key] is not None:
        return float(entry[key])
    for sk in stride_keys:
        if sk in entry and entry[sk] is not None:
            return float(entry[sk])
    return None


def _build_segments(words, max_gap=0.8, max_dur=14.0, max_chars=240):
    segs, cur, prev_end = [], None, None
    enders = (".", "?", "!", "…", "。", "？", "！")

    def flush():
        nonlocal cur
        if cur and cur["_w"]:
            text = " ".join(w["word"].strip() for w in cur["_w"]).strip()
            if text:
                segs.append({"start": cur["start"], "end": cur["end"], "text": text})
        cur = None

    for w in words:
        ws, we = w["start"], w["end"]
        if cur is None:
            cur = {"start": ws if ws is not None else (prev_end or 0.0), "end": we, "_w": [w]}
        else:
            gap = (ws - prev_end) if (ws is not None and prev_end is not None) else 0.0
            dur = (we - cur["start"]) if (we is not None and cur["start"] is not None) else 0.0
            chars = sum(len(x["word"]) for x in cur["_w"])
            if gap > max_gap or dur > max_dur or chars > max_chars:
                flush()
                cur = {"start": ws if ws is not None else (prev_end or 0.0), "end": we, "_w": [w]}
            else:
                cur["_w"].append(w)
                if we is not None:
                    cur["end"] = we
        if w["word"].strip().endswith(enders):
            flush()
        prev_end = we if we is not None else (ws if ws is not None else prev_end)
    flush()
    return segs


def transcribe(audio_path, output_path, model_id, language):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"device={device} | torch={torch.__version__} hip={getattr(torch.version,'hip',None)} "
         f"| model={model_id}")

    asr = nemo_asr.models.ASRModel.from_pretrained(model_id)

    # Belt-and-suspenders: also disable cuda graphs via the decoding config.
    try:
        from omegaconf import open_dict
        dec = asr.cfg.decoding
        if hasattr(dec, "greedy"):
            with open_dict(dec.greedy):
                dec.greedy["use_cuda_graph_decoder"] = False
            asr.change_decoding_strategy(dec)
            _log("cuda_graph_decoder disabled")
    except Exception as e:
        _log(f"decoding cfg note: {e}")

    asr = asr.to(device).eval()
    _log(f"model on {next(asr.parameters()).device}; transcribing {audio_path}")

    t0 = time.time()
    out = asr.transcribe([audio_path], timestamps=True)
    hyp = out[0]
    text = (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()
    ts = getattr(hyp, "timestamp", None) or {}
    raw_words = ts.get("word", []) if isinstance(ts, dict) else []

    words = []
    for w in raw_words:
        start = _seconds(w, "start", ["start_offset"])
        end = _seconds(w, "end", ["end_offset"])
        token = w.get("word") or w.get("char") or ""
        if token:
            words.append({"word": token, "start": round(start, 3) if start is not None else 0.0,
                          "end": round(end, 3) if end is not None else 0.0, "score": 1.0})

    segments = _build_segments(words) if words else ([{"start": 0.0, "end": 0.0, "text": text}] if text else [])

    result = {
        "text": text,
        "language": language or "en",
        "segments": segments,
        "word_segments": words,
        "model": model_id,
        "has_word_timestamps": bool(words),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"done: {len(text)} chars, {len(segments)} segments, {len(words)} words in {time.time()-t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description="NeMo Parakeet transcription (ROCm)")
    p.add_argument("audio_path")
    p.add_argument("output_path")
    p.add_argument("--model-id", default=os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3"))
    p.add_argument("--language", default="en")
    args = p.parse_args()
    if not os.path.exists(args.audio_path):
        _log(f"audio not found: {args.audio_path}")
        sys.exit(1)
    try:
        transcribe(args.audio_path, args.output_path, args.model_id, args.language)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _log(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
