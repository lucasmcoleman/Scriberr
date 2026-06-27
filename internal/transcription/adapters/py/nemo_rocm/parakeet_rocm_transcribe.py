#!/usr/bin/env python3
"""
NVIDIA Parakeet-TDT transcription for Scriberr on AMD ROCm (gfx1151).

Runs NeMo Parakeet purely for inference on the AMD GPU. NeMo's CUDA-graph RNNT/TDT
decoder is disabled (it dlopens libcuda.so.1, absent on AMD); the pure-PyTorch
label-looping decoder runs on ROCm via the torch.cuda (HIP) namespace.

Long audio is processed in fixed windows (chunking) so GPU memory stays bounded
regardless of file length — a single-pass transcribe of an hour-long meeting
OOMs the conformer encoder.

Outputs the WhisperX-compatible JSON shape Scriberr consumes:
  { "text", "language", "segments":[{start,end,text}], "word_segments":[{start,end,word,score}] }
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
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

# Window length (seconds) for long-audio chunking. ~10 min keeps peak GPU memory
# well bounded on the gfx1151 APU while minimizing chunk-boundary effects.
CHUNK_SEC = int(os.environ.get("PARAKEET_CHUNK_SEC", "600"))


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _sec(entry, key, alts):
    if key in entry and entry[key] is not None:
        return float(entry[key])
    for a in alts:
        if a in entry and entry[a] is not None:
            return float(entry[a])
    return None


def _ffprobe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _extract_chunk(src, start, dur, dst):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-t", str(dur),
         "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True, timeout=600)


def _decode_hyp(hyp):
    text = (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()
    ts = getattr(hyp, "timestamp", None) or {}
    raw = ts.get("word", []) if isinstance(ts, dict) else []
    words = []
    for w in raw:
        s = _sec(w, "start", ["start_offset"])
        e = _sec(w, "end", ["end_offset"])
        tok = w.get("word") or w.get("char") or ""
        if tok:
            words.append({"word": tok, "start": s, "end": e})
    return text, words


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
    # Require the GPU. Running NeMo Parakeet on CPU pegs every core and blows up
    # host RAM on long files — fail loudly instead of silently falling back.
    if not torch.cuda.is_available():
        _log("FATAL: GPU not available to torch (torch.cuda.is_available()==False). "
             "Refusing to run Parakeet on CPU. Check /dev/kfd + /dev/dri passthrough and that the GPU isn't saturated.")
        sys.exit(2)
    device = "cuda"
    _log(f"device={device} | torch={torch.__version__} hip={getattr(torch.version,'hip',None)} | model={model_id}")

    asr = nemo_asr.models.ASRModel.from_pretrained(model_id)
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

    total = _ffprobe_duration(audio_path)
    t0 = time.time()
    texts, words = [], []

    if total and total > CHUNK_SEC * 1.25:
        n = math.ceil(total / CHUNK_SEC)
        _log(f"long audio {total/60:.1f} min -> {n} chunks of {CHUNK_SEC}s")
        tmpd = tempfile.mkdtemp(prefix="pk_")
        try:
            for i in range(n):
                start = i * CHUNK_SEC
                dur = min(CHUNK_SEC, total - start)
                cf = os.path.join(tmpd, f"c{i}.wav")
                _extract_chunk(audio_path, start, dur, cf)
                ctext, cwords = _decode_hyp(asr.transcribe([cf], timestamps=True)[0])
                texts.append(ctext)
                for w in cwords:
                    s = (w["start"] if w["start"] is not None else 0.0) + start
                    e = (w["end"] if w["end"] is not None else w["start"] or 0.0) + start
                    words.append({"word": w["word"], "start": round(s, 3), "end": round(e, 3), "score": 1.0})
                _log(f"  chunk {i+1}/{n} [{start:.0f}-{start+dur:.0f}s]: {len(ctext)} chars")
                # Release the chunk's GPU/GTT memory before the next one (bounds
                # peak unified-memory use on the APU).
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
        full_text = " ".join(t for t in texts if t).strip()
    else:
        ctext, cwords = _decode_hyp(asr.transcribe([audio_path], timestamps=True)[0])
        full_text = ctext
        for w in cwords:
            words.append({"word": w["word"], "start": round(w["start"] or 0.0, 3),
                          "end": round(w["end"] or 0.0, 3), "score": 1.0})

    segments = _build_segments(words) if words else ([{"start": 0.0, "end": 0.0, "text": full_text}] if full_text else [])
    result = {
        "text": full_text, "language": language or "en",
        "segments": segments, "word_segments": words,
        "model": model_id, "has_word_timestamps": bool(words),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"done: {len(full_text)} chars, {len(segments)} segments, {len(words)} words in {time.time()-t0:.1f}s")


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
