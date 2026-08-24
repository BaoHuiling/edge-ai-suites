from components.asr.base_asr import BaseASR
from utils import ensure_model
from utils.config_loader import config
from utils.model_download_helper import get_or_download_model_dir
from utils.transformers_overlay import subprocess_env

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


QWEN_MODEL_MAP = {
    # The "-hf" repos are the transformers-native conversions; the plain repos ship
    # weights for Qwen's own inference stack and do not load via AutoProcessor.
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}

# Qwen3-ASR landed in transformers 5.13.0 and safetensors 0.8.0
TRANSFORMERS_VERSION = "5.13.1"
OVERLAY_EXTRA_SPECS = ("safetensors>=0.8.0",)

_WORKER = Path(__file__).resolve().parent / "worker.py"

# The model returns no timestamps, so a chunk yields exactly one segment. ASRComponent
# adds the chunk's own start_time offset, which makes these absolute.
_DEFAULT_WINDOW_SEC = 30.0


class Qwen3ASR(BaseASR):
    """Qwen3-ASR served from a warm subprocess running a side-loaded transformers.

    The subprocess is required, not incidental: ``transformers`` is a ``sys.modules``
    singleton, so importing 5.13 in-process would swap the version out from under the
    VLM tokenizer, OCR, embedding and reranker paths.
    """

    def __init__(self, model_name, device="cpu", revision=None, language=None):
        # language: optional spoken-language hint ("zh", "en", "Chinese", ...).
        # Deliberately a constructor argument and not a config key -- Qwen3-ASR does
        # language identification natively, so the app always lets it auto-detect;
        # only the benchmark, which builds the provider directly, pins a language to
        # keep its runs reproducible.
        self.language = language or None
        key = model_name.lower()
        if key not in QWEN_MODEL_MAP:
            raise ValueError(
                f"Invalid ASR model name {model_name}. "
                f"Supported models are: {list(QWEN_MODEL_MAP.keys())}"
            )
        if device and device.lower() not in ("cpu", ""):
            # No OpenVINO/XPU path for this model yet; fail loudly rather than
            # silently running on CPU while the config claims otherwise.
            raise ValueError(
                f"Qwen3-ASR currently supports device 'CPU' only, got '{device}'."
            )

        repo_id = QWEN_MODEL_MAP[key]
        self.model_name = repo_id

        hub = "hf" if str(getattr(config.models, "model_hub", "huggingface")).lower() \
            == "huggingface" else "ms"
        model_dir = get_or_download_model_dir(
            model=repo_id,
            hub=hub,
            revision=revision,
            local_dir=ensure_model.get_asr_model_path(),
        )

        threads_limit = getattr(config.models.asr, "threads_limit", None)
        threads = threads_limit if threads_limit and threads_limit > 0 else 0
        window_sec = getattr(
            config.audio_preprocessing, "chunk_duration_sec", _DEFAULT_WINDOW_SEC
        ) or _DEFAULT_WINDOW_SEC

        self._lock = threading.Lock()
        self._proc = self._spawn(model_dir, threads, window_sec)

    def _spawn(self, model_dir: str, threads: int, window_sec: float):
        env = subprocess_env(TRANSFORMERS_VERSION, OVERLAY_EXTRA_SPECS)
        # Chinese transcripts travel over the pipe; the Windows console default
        # (GBK) mangles them.
        env["PYTHONIOENCODING"] = "utf-8"

        logger.info(f"Starting Qwen3-ASR worker for {self.model_name} ({model_dir})")
        proc = subprocess.Popen(
            [sys.executable, str(_WORKER),
             "--model-dir", model_dir,
             "--threads", str(threads),
             "--window-sec", str(window_sec)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit, so load errors land in the app log
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        handshake = proc.stdout.readline()
        if not handshake:
            proc.wait(timeout=10)
            raise RuntimeError(
                f"Qwen3-ASR worker exited during startup (code {proc.returncode}); "
                f"see the log above for the traceback."
            )
        if not json.loads(handshake).get("ready"):
            raise RuntimeError(f"Unexpected Qwen3-ASR worker handshake: {handshake!r}")

        logger.info("Qwen3-ASR worker ready")
        return proc

    def transcribe(self, audio_path: str, temperature=0.0) -> dict:
        # temperature is accepted for interface parity with the other providers;
        # decoding here is greedy, matching paraformer's deterministic behaviour.
        empty = {"text": "", "segments": []}
        try:
            with self._lock:
                if self._proc is None or self._proc.poll() is not None:
                    raise RuntimeError("Qwen3-ASR worker is not running")

                request = json.dumps(
                    {"audio": os.path.abspath(audio_path), "language": self.language},
                    ensure_ascii=False,
                )
                self._proc.stdin.write(request + "\n")
                self._proc.stdin.flush()

                line = self._proc.stdout.readline()

            if not line:
                raise RuntimeError("Qwen3-ASR worker closed the pipe")

            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "unknown worker error"))

            text = (response.get("text") or "").strip()
            if not text:
                return empty

            # One segment spanning the chunk. Qwen3-ASR emits no timestamps of its own
            # (that needs the separate Qwen3-ForcedAligner), so granularity here is the
            # chunk the pipeline handed us.
            return {
                "text": text,
                "segments": [{
                    "start": 0.0,
                    "end": float(response.get("duration") or 0.0),
                    "text": text,
                }],
            }

        except Exception as e:
            logger.error(f"[ASR] Qwen3-ASR transcription error: {e}")
            return empty

    def close(self) -> None:
        """Terminate the worker. Safe to call more than once."""
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None or proc.poll() is not None:
                return
            logger.info("Stopping Qwen3-ASR worker...")
            try:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
                proc.wait(timeout=5)
            logger.info("Qwen3-ASR worker stopped")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
