"""Lock-free ring buffer for audio samples."""

import numpy as np
from collections import deque
import threading


class RingBuffer:
    """Thread-safe ring buffer for audio samples.

    Stores mono float32 audio at the configured sample rate.
    Oldest samples are silently evicted on overflow.
    """

    def __init__(self, max_seconds: float = 30.0, sample_rate: int = 16000):
        self._max_samples = int(max_seconds * sample_rate)
        self._sample_rate = sample_rate
        self._buffer = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos = 0
        self._count = 0
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def max_samples(self) -> int:
        return self._max_samples

    def write(self, data: np.ndarray) -> None:
        """Write audio samples into the ring buffer, evicting oldest on overflow."""
        if data.ndim != 1:
            data = data.flatten()

        n = len(data)
        if n == 0:
            return

        with self._lock:
            if n >= self._max_samples:
                # Data larger than entire buffer — keep only the tail
                self._buffer[:] = data[-self._max_samples:]
                self._write_pos = 0
                self._count = self._max_samples
            else:
                # Write in one or two segments (wrap-around)
                end_pos = self._write_pos + n
                if end_pos <= self._max_samples:
                    self._buffer[self._write_pos:end_pos] = data
                else:
                    first = self._max_samples - self._write_pos
                    self._buffer[self._write_pos:] = data[:first]
                    self._buffer[:n - first] = data[first:]

                self._write_pos = end_pos % self._max_samples
                self._count = min(self._count + n, self._max_samples)

    def read_all(self) -> np.ndarray:
        """Read all available samples in chronological order."""
        with self._lock:
            if self._count == 0:
                return np.array([], dtype=np.float32)
            if self._count < self._max_samples:
                return self._buffer[:self._count].copy()
            # Full buffer — read from write_pos (oldest) wrapping around
            return np.concatenate([
                self._buffer[self._write_pos:],
                self._buffer[:self._write_pos],
            ])

    def read_last(self, n_samples: int) -> np.ndarray:
        """Read the most recent n_samples."""
        with self._lock:
            n = min(n_samples, self._count)
            if n == 0:
                return np.array([], dtype=np.float32)
            if self._count < self._max_samples:
                return self._buffer[self._count - n:self._count].copy()
            # Full buffer
            start = (self._write_pos - n) % self._max_samples
            if start < self._write_pos:
                return self._buffer[start:self._write_pos].copy()
            return np.concatenate([
                self._buffer[start:],
                self._buffer[:self._write_pos],
            ])

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_pos = 0
            self._count = 0
            self._buffer[:] = 0.0

    @property
    def available_samples(self) -> int:
        with self._lock:
            return self._count
