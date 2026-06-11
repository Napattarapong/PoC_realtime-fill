"""Typhoon ASR STT backend — local FastConformer model for Thai speech-to-text."""

import logging
import os
import tempfile
import wave
from typing import Optional

import numpy as np

from src.stt.base import STTBase

logger = logging.getLogger(__name__)


def _extract_text(result: dict) -> str:
    """Extract plain text from typhoon_asr result.

    The 'text' field may be a NeMo Hypothesis object or a plain string.
    """
    raw = result.get("text", "")
    if hasattr(raw, "text"):
        return raw.text
    return str(raw)


class TyphoonASRBackend(STTBase):
    """STT backend using Typhoon ASR Real-Time (FastConformer, self-hosted).

    Uses the typhoon-asr pip package for local inference on CPU or GPU.
    No API key required — model auto-downloads from HuggingFace on first use.

    The load() is instant (just stores a function reference).
    The heavy NeMo model loads lazily on the first transcription call.
    """

    def __init__(
        self,
        model_name: str = "scb10x/typhoon-asr-realtime",
        device: str = "auto",
    ):
        super().__init__()
        self._model_name = model_name
        self._device = device
        self._transcribe_fn = None
        self._model_loaded = False

    def load(self) -> None:
        """Prepare the backend (instant — heavy loading deferred to first call)."""
        logger.info(
            f"Typhoon ASR: model={self._model_name}, device={self._device} "
            f"(NeMo loads on first transcription)"
        )
        self._transcribe_fn = None  # will be imported on first use

    def unload(self) -> None:
        """Release model resources."""
        self._transcribe_fn = None
        self._model_loaded = False
        logger.info("Typhoon ASR backend unloaded")

    @property
    def memory_rss_gb(self) -> float:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)

    def _ensure_loaded(self) -> None:
        """Lazily import and warm up the model on first transcription."""
        if self._model_loaded:
            return
        logger.info("Loading Typhoon ASR model (first call)...")
        from typhoon_asr import transcribe
        self._transcribe_fn = transcribe
        self._model_loaded = True
        logger.info("Typhoon ASR model loaded")

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe using Typhoon ASR."""
        self._ensure_loaded()

        # Write to temp WAV — typhoon_asr expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._save_wav(audio, tmp_path)
            result = self._transcribe_fn(
                tmp_path,
                model_name=self._model_name,
                device=self._device,
            )
            text = _extract_text(result)

            proc_time = result.get("processing_time", 0)
            audio_dur = result.get("audio_duration", 0)
            logger.info(
                f"Typhoon ASR: duration={audio_dur:.1f}s, "
                f"processing={proc_time:.2f}s, "
                f"text='{text[:100]}'"
            )
            return text
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000) -> None:
        """Save float32 audio array to a WAV file (16-bit PCM)."""
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
