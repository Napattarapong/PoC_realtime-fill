"""Silero VAD — Voice Activity Detection via the silero_vad package.

Uses the silero_vad pip package (bundles PyTorch model).
Provides:
- Frame-level speech probability scoring
- Speech endpoint detection (configurable silence duration)
- Complete speech segment extraction
- Latency measurement per frame
"""

import logging
import time
from typing import Callable, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Model expects 16kHz mono, 512-sample frames
SAMPLE_RATE = 16000
FRAME_SIZE = 512  # ~32ms per frame at 16kHz


class SileroVAD:
    """Silero Voice Activity Detection.

    Processes 16kHz mono audio in 512-sample frames and fires callbacks
    when speech segments are detected and completed.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/models/silero-vad",
        threshold: float = 0.5,
        silence_duration_ms: int = 100,
        min_speech_duration_ms: int = 250,
        sample_rate: int = SAMPLE_RATE,
        energy_threshold: float = 0.01,
    ):
        self._threshold = float(np.clip(threshold, 0.1, 0.9))
        self._silence_frames = int(silence_duration_ms / (FRAME_SIZE / sample_rate * 1000))
        self._min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
        self._sample_rate = sample_rate
        self._energy_threshold = energy_threshold

        # State
        self._is_speaking = False
        self._speech_buffer: List[np.ndarray] = []
        self._silent_frame_count = 0

        # Latency tracking
        self._frame_times: List[float] = []
        self._rolling_window = 100

        # Load model via silero_vad package
        import silero_vad
        self._model = silero_vad.load_silero_vad()

        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable[[np.ndarray], None]] = None

        logger.info(
            f"Silero VAD initialized: threshold={self._threshold}, "
            f"silence_frames={self._silence_frames}, "
            f"min_speech_samples={self._min_speech_samples}"
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(np.clip(value, 0.1, 0.9))

    @property
    def avg_frame_time_ms(self) -> float:
        """Average frame processing time in ms over the rolling window."""
        if not self._frame_times:
            return 0.0
        recent = self._frame_times[-self._rolling_window:]
        return sum(recent) / len(recent) * 1000

    def on_speech_start(self, callback: Callable[[], None]) -> None:
        """Register callback for speech start event."""
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register callback for speech end event. Receives the complete audio segment."""
        self._on_speech_end = callback

    def _score_frame(self, frame: np.ndarray) -> float:
        """Run a single frame through the Silero model, return speech probability."""
        t0 = time.perf_counter()

        # silero_vad model expects a 1D float32 tensor
        audio_tensor = torch.from_numpy(frame).float()
        prob = self._model(audio_tensor, self._sample_rate).item()

        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > self._rolling_window * 2:
            self._frame_times = self._frame_times[-self._rolling_window:]

        return prob

    def process_frame(self, frame: np.ndarray) -> float:
        """Process a single audio frame and return speech probability.

        Manages internal speech state machine:
        - Detects speech start (transition from silence to speech)
        - Detects speech end (silence duration exceeded)
        - Fires registered callbacks

        Args:
            frame: float32 numpy array of FRAME_SIZE samples (512 at 16kHz)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if len(frame) != FRAME_SIZE:
            logger.warning(
                f"Frame size mismatch: expected {FRAME_SIZE}, got {len(frame)}"
            )
            return 0.0

        prob = self._score_frame(frame)
        rms_energy = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        is_speech = prob >= self._threshold and rms_energy >= self._energy_threshold

        if is_speech:
            if not self._is_speaking:
                # Speech start
                self._is_speaking = True
                self._speech_buffer = []
                self._silent_frame_count = 0
                logger.debug("Speech started")
                if self._on_speech_start:
                    self._on_speech_start()

            self._speech_buffer.append(frame.copy())
            self._silent_frame_count = 0
        else:
            if self._is_speaking:
                self._speech_buffer.append(frame.copy())
                self._silent_frame_count += 1

                if self._silent_frame_count >= self._silence_frames:
                    # Speech end — assemble full segment
                    self._is_speaking = False
                    segment = np.concatenate(self._speech_buffer)

                    # Only fire if segment meets minimum duration
                    if len(segment) >= self._min_speech_samples:
                        logger.debug(
                            f"Speech ended: {len(segment)} samples "
                            f"({len(segment)/self._sample_rate*1000:.0f}ms)"
                        )
                        if self._on_speech_end:
                            self._on_speech_end(segment)
                    else:
                        logger.debug(
                            f"Speech segment too short ({len(segment)} samples), ignoring"
                        )

                    self._speech_buffer = []
                    self._silent_frame_count = 0

        return prob

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._speech_buffer = []
        self._silent_frame_count = 0
        logger.debug("VAD state reset")
