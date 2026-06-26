#!/usr/bin/env python3
"""
Whisper (Hugging Face Transformers) transcription for Scriberr.

A pure-PyTorch ASR engine: runs OpenAI Whisper weights through the
`transformers` automatic-speech-recognition pipeline with SDPA attention.
Because it is plain ATen/SDPA (no CTranslate2, no torchvision, no custom CUDA
kernels) it runs unmodified on AMD ROCm/HIP GPUs (which present through the
torch.cuda namespace) as well as NVIDIA CUDA and CPU.

Output matches the WhisperX JSON shape Scriberr already consumes:
  { "text", "language", "segments":[{start,end,text}], "word_segments":[{start,end,word,score}] }
"""

import argparse
import json
import sys
import os

import torch
from transformers import pipeline


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pick_device_dtype():
    """ROCm reports availability through torch.cuda. Use fp16 on GPU (gfx1151 has
    known bf16 numerical bugs), fp32 on CPU."""
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "GPU"
        _log(f"GPU detected: {name} | torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} cuda={getattr(torch.version, 'cuda', None)}")
        return "cuda:0", torch.float16
    _log(f"No GPU detected; running on CPU | torch={torch.__version__}")
    return "cpu", torch.float32


def _build_segments(words, max_gap=0.8, max_dur=12.0, max_chars=240):
    """Group word-level chunks into readable segments.

    A new segment starts when the previous word ended a sentence (. ? ! …),
    when the silent gap to the next word exceeds max_gap, or when the running
    segment grows past max_dur seconds / max_chars characters.
    """
    segments = []
    cur = None
    enders = (".", "?", "!", "…", "。", "？", "！")

    def flush():
        nonlocal cur
        if cur and cur["_words"]:
            text = "".join(w["word"] for w in cur["_words"]).strip()
            if text:
                segments.append({
                    "start": cur["start"],
                    "end": cur["end"],
                    "text": text,
                })
        cur = None

    prev_end = None
    for w in words:
        ws, we = w["start"], w["end"]
        if cur is None:
            cur = {"start": ws if ws is not None else (prev_end or 0.0), "end": we, "_words": [w]}
        else:
            gap = (ws - prev_end) if (ws is not None and prev_end is not None) else 0.0
            dur = (we - cur["start"]) if (we is not None and cur["start"] is not None) else 0.0
            chars = sum(len(x["word"]) for x in cur["_words"])
            if gap > max_gap or dur > max_dur or chars > max_chars:
                flush()
                cur = {"start": ws if ws is not None else (prev_end or 0.0), "end": we, "_words": [w]}
            else:
                cur["_words"].append(w)
                if we is not None:
                    cur["end"] = we
        # sentence boundary -> close segment after this word
        stripped = w["word"].strip()
        if stripped.endswith(enders):
            flush()
        if we is not None:
            prev_end = we
        elif ws is not None:
            prev_end = ws
    flush()
    return segments


def transcribe(audio_path, output_path, model_id, language, batch_size, chunk_length):
    device, dtype = _pick_device_dtype()

    _log(f"Loading ASR pipeline: model={model_id} dtype={dtype} device={device} (attn=sdpa)")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=dtype,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    gen_kwargs = {"task": "transcribe"}
    if language and language.lower() != "auto":
        gen_kwargs["language"] = language

    _log(f"Transcribing {audio_path} (chunk_length_s={chunk_length}, batch_size={batch_size})")
    out = pipe(
        audio_path,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        return_timestamps="word",
        generate_kwargs=gen_kwargs,
    )

    full_text = (out.get("text") or "").strip()
    chunks = out.get("chunks") or []

    words = []
    last_end = 0.0
    for ch in chunks:
        token = ch.get("text", "")
        ts = ch.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        # transformers occasionally emits None (esp. the final word's end) -> backfill
        if start is None:
            start = last_end
        if end is None:
            end = start
        if end < start:
            end = start
        words.append({
            "word": token,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "score": 1.0,
        })
        last_end = end

    segments = _build_segments(words)
    if not segments and full_text:
        # No usable timestamps -> single segment spanning the whole clip
        segments = [{"start": 0.0, "end": last_end, "text": full_text}]

    result = {
        "text": full_text,
        "language": (language if (language and language.lower() != "auto") else "en"),
        "segments": segments,
        "word_segments": words,
        "model": model_id,
        "has_word_timestamps": bool(words),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _log(f"Done: {len(full_text)} chars, {len(segments)} segments, {len(words)} words -> {output_path}")
    return result


def main():
    p = argparse.ArgumentParser(description="Whisper (HF Transformers) transcription for Scriberr")
    p.add_argument("audio_path", help="Path to input audio file")
    p.add_argument("output_path", help="Path to output JSON file")
    p.add_argument("--model-id", default=os.environ.get("WHISPER_HF_MODEL", "openai/whisper-large-v3-turbo"))
    p.add_argument("--language", default="auto", help="Language code or 'auto'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-length", type=int, default=30)
    args = p.parse_args()

    if not os.path.exists(args.audio_path):
        _log(f"Error: audio file not found: {args.audio_path}")
        sys.exit(1)

    try:
        transcribe(
            audio_path=args.audio_path,
            output_path=args.output_path,
            model_id=args.model_id,
            language=args.language,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
        )
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _log(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
