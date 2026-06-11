"""Async audio capture from macOS input devices."""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd

from src.audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


def check_microphone_permission() -> bool:
    """Check if macOS microphone permission has been granted.

    Returns True if permission is available or not applicable (non-macOS).
    On macOS, attempts a brief recording to trigger/check permission.
    """
    if sys.platform != "darwin":
        return True

    try:
        # Attempt a tiny recording to check permission
        test = sd.rec(1, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        if test is not None:
            return True
    except Exception as e:
        error_msg = str(e).lower()
        if "permission" in error_msg or "access" in error_msg or "denied" in error_msg:
            logger.error(
                "Microphone permission denied.\n"
                "Grant access in: System Settings → Privacy & Security → Microphone\n"
                "Then restart this application."
            )
            return False
        # Other error (no device, etc.) — let it surface later
        raise
    return True


def list_input_devices() -> List[Dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append({
                "index": i,
                "name": dev["name"],
                "sample_rate": int(dev["default_samplerate"]),
                "channels": dev["max_input_channels"],
            })
    return input_devices


def auto_detect_device(target_sample_rate: int = 16000) -> Optional[int]:
    """Auto-detect the best input device.

    Prefers the macOS default input device. Returns device index or None.
    """
    try:
        default = sd.default.device
        if default[0] is not None and default[0] >= 0:
            dev_info = sd.query_devices(default[0])
            logger.info(f"Auto-detected input device: {dev_info['name']} "
                        f"(sr={dev_info['default_samplerate']}Hz)")
            return default[0]
    except Exception as e:
        logger.warning(f"Could not auto-detect input device: {e}")
    return None


class AudioCapture:
    """Async audio capture that writes into a ring buffer.

    Captures 16 kHz mono float32 audio using sounddevice (PortAudio/CoreAudio
    on macOS). Audio is written into a RingBuffer for downstream consumers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 512,
        ring_buffer_seconds: float = 30.0,
        input_device: Optional[int] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._ring_buffer = RingBuffer(
            max_seconds=ring_buffer_seconds,
            sample_rate=sample_rate,
        )
        self._input_device = input_device
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._capture_sample_rate: float = float(sample_rate)

    @property
    def ring_buffer(self) -> RingBuffer:
        return self._ring_buffer

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def actual_sample_rate(self) -> float:
        """The actual sample rate of the capture stream (may differ from target)."""
        return self._capture_sample_rate

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status: sd.CallbackFlags) -> None:
        """sounddevice callback — called from audio thread."""
        if status:
            logger.debug(f"Audio capture status: {status}")
        # indata shape: (frames, channels) — take first channel
        mono = indata[:, 0].astype(np.float32)
        self._ring_buffer.write(mono)

    async def start(self) -> None:
        """Start capturing audio."""
        if self._running:
            return

        # Check microphone permission on macOS
        if not check_microphone_permission():
            raise RuntimeError(
                "Microphone permission not granted. "
                "Enable in System Settings → Privacy & Security → Microphone"
            )

        device = self._input_device
        if device is None:
            device = auto_detect_device(self._sample_rate)

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._block_size,
                device=device,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._capture_sample_rate = self._stream.samplerate
            self._running = True
            logger.info(
                f"Audio capture started: sr={self._capture_sample_rate}Hz, "
                f"block={self._block_size}, device={device}"
            )
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise

    async def stop(self) -> None:
        """Stop capturing audio."""
        if not self._running:
            return
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    def get_latest_frames(self, n_frames: int) -> np.ndarray:
        """Get the most recent n_frames from the ring buffer.

        Returns float32 numpy array at the capture sample rate.
        """
        return self._ring_buffer.read_last(n_frames)
