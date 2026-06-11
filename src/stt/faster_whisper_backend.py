"""Faster-Whisper STT backend — CTranslate2 with MPS or CPU on Apple Silicon."""

import logging
from typing import Optional

import numpy as np

from src.stt.base import STTBase

logger = logging.getLogger(__name__)


class FasterWhisperBackend(STTBase):
    """STT backend using Faster-Whisper via CTranslate2.

    Supports MPS (Metal) and CPU backends on Apple Silicon.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        super().__init__()
        self._model_size = model_size
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._rss_gb = 0.0

    def load(self) -> None:
        """Load the Faster-Whisper model."""
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Faster-Whisper: model={self._model_size}, "
            f"device={self._device}, compute_type={self._compute_type}"
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Faster-Whisper model loaded")

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("Faster-Whisper backend unloaded")

    @property
    def memory_rss_gb(self) -> float:
        import psutil
        self._rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        return self._rss_gb

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe using Faster-Whisper."""
        if self._model is None:
            raise RuntimeError("Faster-Whisper not loaded. Call load() first.")

        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,          # Fastest setting
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,  # below this = no speech
            log_prob_threshold=-1.0,  # accept all segments
        )

        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list if seg.text.strip())
        logger.info(
            f"Faster-Whisper: lang={info.language} ({info.language_probability:.2f}), "
            f"duration={info.duration:.1f}s, segments={len(seg_list)}, "
            f"text='{text[:100]}'"
        )
        return text
