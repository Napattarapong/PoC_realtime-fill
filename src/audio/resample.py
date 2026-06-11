"""Audio resampling utilities."""

import numpy as np
import librosa


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio: float32 numpy array (mono)
        orig_sr: original sample rate
        target_sr: target sample rate

    Returns:
        Resampled float32 numpy array at target_sr
    """
    if orig_sr == target_sr:
        return audio
    if len(audio) == 0:
        return audio

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.astype(np.float32)
