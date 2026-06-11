"""whisper.cpp STT backend — native Apple Silicon / Metal acceleration.

Uses whisper.cpp via subprocess for maximum compatibility.
Falls back to pywhispercpp if available.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from src.stt.base import STTBase

logger = logging.getLogger(__name__)


def _find_whisper_cpp() -> str:
    """Find the whisper.cpp binary on the system."""
    # Check common names
    for name in ["whisper-cpp", "whisper", "main"]:
        path = shutil.which(name)
        if path:
            return path

    # Check common Homebrew paths
    for p in [
        "/opt/homebrew/bin/whisper-cpp",
        "/usr/local/bin/whisper-cpp",
        "/opt/homebrew/bin/main",
    ]:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "whisper.cpp not found. Install with: brew install whisper-cpp"
    )


def _find_model(model_size: str, cache_dir: str) -> str:
    """Find or download the whisper.cpp model."""
    model_dir = os.path.expanduser(cache_dir)
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"ggml-{model_size}.bin"
    model_path = os.path.join(model_dir, model_name)

    if os.path.exists(model_path):
        return model_path

    # Download using whisper.cpp's download script convention
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_name}"
    logger.info(f"Downloading whisper model: {model_name}...")
    import urllib.request
    urllib.request.urlretrieve(url, model_path)
    logger.info(f"Model saved to: {model_path}")
    return model_path


class WhisperCppBackend(STTBase):
    """STT backend using whisper.cpp with Metal acceleration.

    Runs whisper.cpp as a subprocess for stability. Writes audio to a
    temporary WAV file and parses JSON output.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        cache_dir: str = ".cache/models/whisper",
    ):
        super().__init__()
        self._model_size = model_size
        self._language = language
        self._cache_dir = cache_dir
        self._bin_path: Optional[str] = None
        self._model_path: Optional[str] = None
        self._loaded = False
        self._rss_gb = 0.0

    def load(self) -> None:
        """Locate binary and download model if needed."""
        self._bin_path = _find_whisper_cpp()
        self._model_path = _find_model(self._model_size, self._cache_dir)
        self._loaded = True
        logger.info(
            f"whisper.cpp backend loaded: bin={self._bin_path}, "
            f"model={self._model_path}"
        )

    def unload(self) -> None:
        """No persistent resources to free (subprocess model)."""
        self._loaded = False
        logger.info("whisper.cpp backend unloaded")

    @property
    def memory_rss_gb(self) -> float:
        return self._rss_gb

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe via whisper.cpp subprocess."""
        if not self._loaded:
            raise RuntimeError("whisper.cpp backend not loaded. Call load() first.")

        # Write audio to temp WAV file
        import struct
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                # Convert float32 to int16
                audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            # Build command
            cmd = [
                self._bin_path,
                "-m", self._model_path,
                "-f", wav_path,
                "--output-json",
                "--no-timestamps",
            ]
            if self._language:
                cmd.extend(["-l", self._language])

            # whisper.cpp writes JSON to {wav_path}.json when --output-json is set
            json_path = wav_path + ".json"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse JSON output
            if os.path.exists(json_path):
                with open(json_path) as jf:
                    data = json.load(jf)
                text = data.get("transcription", [{}])[0].get("text", "")
                return text

            # Fallback: parse stdout
            for line in result.stdout.splitlines():
                if line.strip() and not line.startswith("["):
                    return line

            return ""

        finally:
            # Cleanup temp files
            for p in [wav_path, wav_path + ".json"]:
                if os.path.exists(p):
                    os.unlink(p)
