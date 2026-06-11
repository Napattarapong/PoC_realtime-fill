"""Speech-to-Text module with pluggable backends."""

from typing import Optional

from src.stt.base import STTBase
from src.stt.whisper_cpp_backend import WhisperCppBackend
from src.stt.faster_whisper_backend import FasterWhisperBackend
from src.stt.typhoon_asr_backend import TyphoonASRBackend


def create_stt_backend(
    backend: str = "whisper_cpp",
    model_size: str = "base",
    language: Optional[str] = None,
    cache_dir: str = ".cache/models/whisper",
    device: str = "cpu",
    compute_type: str = "int8",
    model_name: str = "scb10x/typhoon-asr-realtime",
) -> STTBase:
    """Factory function to create an STT backend.

    Args:
        backend: "whisper_cpp", "faster_whisper", or "typhoon_asr"
        model_size: Model size for whisper backends (tiny, base, small, medium)
        language: Language code (None for auto-detect, "th" for Thai, "en" for English)
        cache_dir: Directory for model cache
        device: Device for faster-whisper ("cpu" or "mps") or typhoon_asr ("auto", "cpu", "cuda")
        compute_type: Compute type for faster-whisper ("int8", "float16", etc.)
        model_name: HuggingFace model ID for typhoon_asr backend

    Returns:
        Initialized STTBase instance (call .load() before use)
    """
    if backend == "whisper_cpp":
        return WhisperCppBackend(
            model_size=model_size,
            language=language,
            cache_dir=cache_dir,
        )
    elif backend == "faster_whisper":
        return FasterWhisperBackend(
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
        )
    elif backend == "typhoon_asr":
        return TyphoonASRBackend(
            model_name=model_name,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown STT backend: {backend}. "
            f"Supported: 'whisper_cpp', 'faster_whisper', 'typhoon_asr'"
        )


__all__ = ["STTBase", "WhisperCppBackend", "FasterWhisperBackend", "TyphoonASRBackend", "create_stt_backend"]
