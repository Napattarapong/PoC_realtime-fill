"""Abstract base class for STT backends."""

import abc
import logging
import time
from typing import Optional

import numpy as np

from src.audio.resample import resample

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


class STTBase(abc.ABC):
    """Base class for speech-to-text backends.

    Handles common concerns:
    - Resampling to 16kHz if needed
    - Empty / short audio handling
    - Latency measurement
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self._target_sr = sample_rate
        self._last_latency_ms: float = 0.0

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @abc.abstractmethod
    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe 16kHz float32 mono audio. Must be implemented by subclasses."""
        ...

    @abc.abstractmethod
    def load(self) -> None:
        """Load the STT model into memory."""
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @property
    @abc.abstractmethod
    def memory_rss_gb(self) -> float:
        """Current RSS memory in GB."""
        ...

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> str:
        """Transcribe audio to text.

        Args:
            audio: float32 numpy array (mono)
            sample_rate: sample rate of the audio. If None, assumes 16kHz.

        Returns:
            Transcribed text string. Empty string if no speech detected.
        """
        # Handle empty audio
        if audio is None or len(audio) == 0:
            logger.debug("Empty audio segment, returning empty string")
            return ""

        # Resample if needed
        sr = sample_rate or self._target_sr
        if sr != self._target_sr:
            audio = resample(audio, orig_sr=sr, target_sr=self._target_sr)

        # Handle very short audio (< 100ms at 16kHz = 1600 samples)
        if len(audio) < 1600:
            logger.debug(f"Very short audio ({len(audio)} samples), attempting transcription")

        t0 = time.perf_counter()
        try:
            text = self._transcribe_audio(audio)
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return ""

        elapsed = (time.perf_counter() - t0) * 1000
        self._last_latency_ms = elapsed
        logger.debug(f"STT latency: {elapsed:.1f}ms for {len(audio)/self._target_sr:.1f}s audio")

        return text.strip()
