"""Standalone Qwen3-ASR inference worker.

Runs as a subprocess with a side-loaded transformers build on PYTHONPATH (see
``utils/transformers_overlay.py``) because Qwen3-ASR needs a newer transformers than
the one the rest of the app is pinned to. Like ``convert_worker.py`` it imports
**nothing** from the application -- it must stay loadable under a transformers version
the app itself cannot import.

Protocol: newline-delimited JSON on stdin/stdout, one response per request.

    <- {"ready": true}                                   (once, after model load)
    -> {"audio": "<path>", "language": "zh"}
    <- {"ok": true, "text": "...", "windows": 3}
    -> {"cmd": "shutdown"}

Errors are reported as ``{"ok": false, "error": "..."}`` rather than raised, so one bad
chunk does not take down a warm model.
"""
import argparse
import json
import sys

# Model weights load once at startup; every subsequent request reuses them.
_MODEL = None
_PROCESSOR = None

TARGET_SR = 16000

# Generation budget per window. Natural Chinese speech runs ~4-6 characters/second and
# Qwen3-ASR emits roughly one token per character, so 12 tokens/second is ~2x headroom
# over the fastest realistic speech while still bounding a runaway decode.
_TOKENS_PER_SEC = 12
_MIN_NEW_TOKENS = 64
_MAX_NEW_TOKENS = 1024


def _load(model_dir: str, threads: int | None):
    """Load processor + model once, into module globals."""
    global _MODEL, _PROCESSOR
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    if threads and threads > 0:
        torch.set_num_threads(threads)

    _PROCESSOR = AutoProcessor.from_pretrained(model_dir)
    # float32: this is a CPU deployment, where torch's bfloat16 kernels are slower than
    # fp32 for a model this size.
    _MODEL = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.float32)
    _MODEL.eval()


def _read_audio(path: str):
    """Decode any supported container to mono 16 kHz float32.

    Uses librosa rather than letting the processor resolve the path itself: the
    processor's default decoder is torchcodec, whose native libraries fail to load in
    this venv.
    """
    import librosa
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio


def _split_windows(audio, window_sec: float, search_sec: float = 1.0):
    """Split audio into <= window_sec pieces, cutting at the quietest nearby sample.

    The pipeline normally hands this worker chunks that are already silence-aligned
    (``chunk_by_silence``), so this only bites when the whole file arrives in one piece
    (``audio_preprocessing.chunking: false``). Cutting at the local energy minimum
    instead of a fixed offset keeps the split off the middle of a word.
    """
    import numpy as np

    window = int(window_sec * TARGET_SR)
    if window <= 0 or len(audio) <= window:
        return [audio]

    search = int(search_sec * TARGET_SR)
    frame = int(0.02 * TARGET_SR)  # 20 ms energy frames

    pieces = []
    start = 0
    while start < len(audio):
        ideal_end = start + window
        if ideal_end >= len(audio):
            pieces.append(audio[start:])
            break

        lo = max(start + frame, ideal_end - search)
        hi = min(len(audio) - frame, ideal_end + search)
        if hi <= lo:
            cut = ideal_end
        else:
            region = audio[lo:hi]
            n_frames = len(region) // frame
            if n_frames < 1:
                cut = ideal_end
            else:
                frames = region[:n_frames * frame].reshape(n_frames, frame)
                energy = np.abs(frames).mean(axis=1)
                cut = lo + int(energy.argmin()) * frame

        pieces.append(audio[start:cut])
        start = cut

    return [p for p in pieces if len(p) > 0]


def _transcribe_window(audio, language: str | None) -> str:
    import torch

    duration = len(audio) / TARGET_SR
    max_new = int(min(_MAX_NEW_TOKENS, max(_MIN_NEW_TOKENS, duration * _TOKENS_PER_SEC)))

    inputs = _PROCESSOR.apply_transcription_request(
        audio=audio, language=language
    ).to(_MODEL.device, _MODEL.dtype)

    with torch.no_grad():
        output_ids = _MODEL.generate(**inputs, max_new_tokens=max_new, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return _PROCESSOR.decode(generated, return_format="transcription_only")[0]


def _handle(request: dict, window_sec: float) -> dict:
    audio = _read_audio(request["audio"])
    language = request.get("language") or None

    windows = _split_windows(audio, window_sec)
    texts = [_transcribe_window(w, language) for w in windows]
    text = "".join(t.strip() for t in texts if t and t.strip())

    return {"ok": True, "text": text, "windows": len(windows),
            "duration": round(len(audio) / TARGET_SR, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--window-sec", type=float, default=30.0)
    args = parser.parse_args()

    _load(args.model_dir, args.threads)
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad request: {exc}"}), flush=True)
            continue

        if request.get("cmd") == "shutdown":
            break

        try:
            response = _handle(request, args.window_sec)
        except Exception as exc:  # never let one chunk kill a warm model
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
