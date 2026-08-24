"""Compare ASR providers on accuracy, speed and resource usage.

Drives the provider classes directly rather than going through ``/transcribe`` so the
numbers measure the model, not HTTP, upload, diarization or storage. Audio is
normalized to mono 16 kHz WAV once up front so every provider sees identical input.

Run from smart-classroom/ with the project venv:

    python -m components.tests.bench_asr \\
        --audio "C:/path/math_10mins.m4a" --ref "C:/path/ground_truth.txt" \\
        --audio "C:/path/video-0723-1.wav" \\
        --variants funasr/paraformer-zh qwen/Qwen3-ASR-0.6B qwen/Qwen3-ASR-1.7B \\
        --outdir results/

Every transcript is written to ``--outdir`` for manual inspection, which is the only
way to judge an audio file that has no ground truth.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

import psutil

from utils.config_loader import config

TARGET_SR = 16000


# --------------------------------------------------------------------------- audio

def normalize_audio(source: Path, target: Path) -> None:
    """Decode to mono 16 kHz WAV so provider-side decoding differences cancel out."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-ac", "1", "-ar", str(TARGET_SR), str(target)],
        check=True,
    )


def audio_duration(path: Path) -> float:
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# ----------------------------------------------------------------------- accuracy

def normalize_text(text: str) -> str:
    """Reduce a transcript to the comparable character stream.

    Drops speaker prefixes ("老师：" / "学生：" / "TEACHER:"), timestamps, all
    whitespace and all Unicode punctuation. Punctuation stripping matches
    ``ASRComponent._meaningful_char_count`` (components/asr_component.py) -- ASR
    punctuation is a formatting choice, not a recognition result, and counting it
    would penalise models for a stylistic difference.
    """
    import re
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\[[\d\.\s\-]+\]\s*", "", line)      # [12-34] timestamps
        line = re.sub(r"^\s*(老师|学生|教师|说话人[_\d]*|TEACHER|STUDENT|SPEAKER[_\d]*)\s*[:：]\s*",
                      "", line, flags=re.IGNORECASE)
        lines.append(line)
    joined = "".join(lines)
    return "".join(
        c for c in joined
        if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate = Levenshtein(ref, hyp) / len(ref).

    Implemented here on purpose: jiwer is not in the venv and this benchmark is not
    worth a new runtime dependency. Two-row DP, so memory is O(len(hyp)).
    """
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return float("nan")

    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (r != h)  # substitution
            ))
        previous = current
    return previous[-1] / len(ref)


# ---------------------------------------------------------------------- resources

class ResourceSampler:
    """Peak RSS and total CPU time for this process plus any worker children.

    Children matter: Qwen3-ASR holds its weights in a subprocess, so sampling only
    the parent would report near-zero memory for it.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.cpu_seconds = 0.0
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self._cpu_start = self._cpu_total()

    def _procs(self):
        yield self._proc
        try:
            yield from self._proc.children(recursive=True)
        except psutil.Error:
            pass

    def _cpu_total(self) -> float:
        total = 0.0
        for p in self._procs():
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total

    def _run(self):
        while not self._stop.wait(self.interval):
            rss = 0.0
            for p in self._procs():
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak_rss_mb = max(self.peak_rss_mb, rss / 1024 / 1024)
            self.cpu_seconds = max(self.cpu_seconds, self._cpu_total() - self._cpu_start)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self.cpu_seconds = max(self.cpu_seconds, self._cpu_total() - self._cpu_start)


# ---------------------------------------------------------------------- providers

def build_provider(provider: str, name: str, language: str):
    """Point the shared config at this variant, then construct it.

    Both providers resolve their weights through ``ensure_model.get_asr_model_path()``,
    which reads ``config.models.asr``, so the config has to be set before construction.
    """
    config.models.asr.provider = provider
    config.models.asr.name = name

    if provider == "funasr":
        from components.asr.funasr.paraformer import Paraformer
        # paraformer-zh is Chinese-only and takes no language input.
        return Paraformer(name, config.models.asr.device.lower(), "v2.0.4")
    if provider == "qwen":
        from components.asr.qwen.qwen3_asr import Qwen3ASR
        # Passed explicitly so a run is reproducible from its command line alone.
        # The app itself never sets this -- it always lets the model auto-detect.
        hint = None if language.lower() in ("", "auto", "none") else language
        return Qwen3ASR(name, "cpu", None, language=hint)
    raise ValueError(f"Unsupported provider for benchmarking: {provider}")


def run_variant(provider: str, name: str, audio: Path, duration: float,
                language: str) -> dict:
    load_start = time.perf_counter()
    processor = build_provider(provider, name, language)
    load_sec = time.perf_counter() - load_start

    try:
        with ResourceSampler() as sampler:
            infer_start = time.perf_counter()
            result = processor.transcribe(str(audio), temperature=0.0)
            infer_sec = time.perf_counter() - infer_start
    finally:
        closer = getattr(processor, "close", None)
        if callable(closer):
            closer()

    text = (result or {}).get("text", "")
    return {
        "variant": f"{provider}/{name}",
        "load_sec": round(load_sec, 2),
        "infer_sec": round(infer_sec, 2),
        "rtf": round(duration / infer_sec, 2) if infer_sec else float("nan"),
        "peak_rss_mb": round(sampler.peak_rss_mb, 1),
        "cpu_sec": round(sampler.cpu_seconds, 1),
        "segments": len((result or {}).get("segments", [])),
        "chars": len(normalize_text(text)),
        "text": text,
    }


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="append", required=True,
                        help="audio file; repeatable")
    parser.add_argument("--ref", action="append", default=[],
                        help="ground truth for the audio at the same position; "
                             "pass '' or omit to skip CER for that file")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="provider/name pairs, e.g. funasr/paraformer-zh")
    parser.add_argument("--outdir", default="results/asr_bench")
    parser.add_argument("--language", default="auto",
                        help="spoken-language hint for providers that accept one "
                             "(Qwen3-ASR); 'auto' lets the model detect it")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "_normalized"
    workdir.mkdir(exist_ok=True)

    rows = []
    for index, audio_arg in enumerate(args.audio):
        source = Path(audio_arg)
        stem = source.stem
        normalized = workdir / f"{stem}.wav"
        if not normalized.exists():
            print(f"[prep] normalizing {source.name} -> {normalized.name}", flush=True)
            normalize_audio(source, normalized)
        duration = audio_duration(normalized)

        ref_path = args.ref[index] if index < len(args.ref) else ""
        reference = Path(ref_path).read_text(encoding="utf-8") if ref_path else None
        print(f"\n=== {stem}  ({duration:.1f}s, "
              f"ref={'yes' if reference else 'none'}) ===", flush=True)

        for variant in args.variants:
            provider, _, name = variant.partition("/")
            print(f"[run] {variant} ...", flush=True)
            try:
                row = run_variant(provider, name, normalized, duration, args.language)
            except Exception as exc:
                print(f"[FAIL] {variant}: {type(exc).__name__}: {exc}", flush=True)
                rows.append({"audio": stem, "variant": variant,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue

            row["audio"] = stem
            row["duration_sec"] = round(duration, 1)
            row["cer"] = round(cer(reference, row["text"]), 4) if reference else ""
            row["language"] = args.language

            transcript = outdir / f"{stem}__{provider}-{name}.txt"
            transcript.write_text(row.pop("text"), encoding="utf-8")
            print(f"      {row['infer_sec']}s  RTF {row['rtf']}x  "
                  f"peak {row['peak_rss_mb']}MB  CER {row['cer'] or 'n/a'}  "
                  f"-> {transcript.name}", flush=True)
            rows.append(row)

    fields = ["audio", "duration_sec", "variant", "language", "cer", "load_sec",
              "infer_sec", "rtf", "peak_rss_mb", "cpu_sec", "segments", "chars",
              "error"]
    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults -> {csv_path}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
