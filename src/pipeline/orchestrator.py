"""Pipeline orchestrator — connects Audio → VAD → STT → LLM.

Manages the full lifecycle:
- Ordered startup: LLM warmup → STT load → VAD load → audio capture
- Ordered shutdown (reverse) on SIGINT/SIGTERM or component failure
- Push-to-talk and always-listening modes
- Streaming LLM output to stdout or callback
- Per-request latency logging
"""

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, List, Optional

import numpy as np

from src.audio.capture import AudioCapture
from src.audio.ring_buffer import RingBuffer
from src.stt import STTBase, create_stt_backend
from src.vad.silero_vad import SileroVAD
from src.llm.typhoon_engine import LLMRequest, get_schema
from src.llm.typhoon_api_engine import TyphoonEngine

logger = logging.getLogger(__name__)


@dataclass
class LatencyLog:
    """Per-request latency breakdown."""
    vad_end_ms: float = 0.0
    stt_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    total_e2e_ms: float = 0.0
    audio_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "vad_end_ms": round(self.vad_end_ms, 1),
            "stt_latency_ms": round(self.stt_latency_ms, 1),
            "llm_ttft_ms": round(self.llm_ttft_ms, 1),
            "total_e2e_ms": round(self.total_e2e_ms, 1),
            "audio_duration_ms": round(self.audio_duration_ms, 1),
        }


class PipelineOrchestrator:
    """Main orchestrator for the voice pipeline.

    Coordinates all components and manages the audio → VAD → STT → LLM flow.

    Meeting assistant mode: listens continuously, transcribes everything,
    keeps conversation history, and only responds to questions.
    """

    def __init__(
        self,
        config: dict,
        token_callback: Optional[Callable[[str], None]] = None,
    ):
        self._config = config
        self._token_callback = token_callback
        self._running = False
        self._components_started: List[str] = []

        # Components (initialized during startup)
        self._llm: Optional[TyphoonEngine] = None
        self._stt: Optional[STTBase] = None
        self._vad: Optional[SileroVAD] = None
        self._capture: Optional[AudioCapture] = None

        # Pipeline state
        self._pending_audio: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._latency_logs: List[LatencyLog] = []

        # Meeting assistant state
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history: int = self._config.get("pipeline", {}).get("max_history", 50)
        self._is_responding = False  # prevent overlapping LLM calls

    @property
    def is_running(self) -> bool:
        return self._running

    def _init_llm(self) -> TyphoonEngine:
        llm_cfg = self._config.get("llm", {})
        return TyphoonEngine(
            api_key=llm_cfg.get("api_key"),
            model=llm_cfg.get("model", "typhoon-v2.5-30b-a3b-instruct"),
            base_url=llm_cfg.get("base_url", "https://api.opentyphoon.ai/v1"),
            system_prompt=llm_cfg.get("system_prompt", ""),
            generation_params=llm_cfg.get("generation", {}),
            # Legacy local params (ignored, kept for config compatibility)
            model_path=llm_cfg.get("model_path", ""),
            n_gpu_layers=llm_cfg.get("n_gpu_layers", -1),
            n_ctx=llm_cfg.get("n_ctx", 1024),
        )

    def _init_stt(self) -> STTBase:
        stt_cfg = self._config.get("stt", {})
        return create_stt_backend(
            backend=stt_cfg.get("backend", "whisper_cpp"),
            model_size=stt_cfg.get("model_size", "base"),
            language=stt_cfg.get("language"),
            cache_dir=stt_cfg.get("cache_dir", ".cache/models/whisper"),
            device=stt_cfg.get("device", "auto"),
            model_name=stt_cfg.get("model_name", "scb10x/typhoon-asr-realtime"),
        )

    def _init_vad(self) -> SileroVAD:
        vad_cfg = self._config.get("vad", {})
        return SileroVAD(
            cache_dir=vad_cfg.get("cache_dir", ".cache/models/silero-vad"),
            threshold=vad_cfg.get("threshold", 0.5),
            silence_duration_ms=vad_cfg.get("silence_duration_ms", 100),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 250),
            energy_threshold=vad_cfg.get("energy_threshold", 0.01),
        )

    @staticmethod
    def _segment_thai(text: str) -> str:
        """Segment Thai words with spaces using pythainlp.

        Only segments Thai characters — leaves English/punctuation untouched.
        Falls back to raw text if pythainlp is not available.
        """
        if not text:
            return text
        try:
            from pythainlp.tokenize import word_tokenize
            return " ".join(word_tokenize(text, engine="newmm"))
        except ImportError:
            return text

    def _init_capture(self) -> AudioCapture:
        audio_cfg = self._config.get("audio", {})
        return AudioCapture(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
            block_size=audio_cfg.get("block_size", 512),
            ring_buffer_seconds=audio_cfg.get("ring_buffer_seconds", 30),
            input_device=audio_cfg.get("input_device"),
        )

    async def start(self) -> None:
        """Start all components in order: LLM → STT → VAD → Audio."""
        logger.info("Starting pipeline...")

        try:
            # 1. LLM (slowest to start, do first)
            logger.info("[1/4] Loading Typhoon LLM...")
            self._llm = self._init_llm()
            self._llm.load()
            self._llm.warmup_prompt_cache()
            self._components_started.append("llm")

            # 2. STT
            logger.info("[2/4] Loading STT backend...")
            self._stt = self._init_stt()
            self._stt.load()
            self._components_started.append("stt")

            # 3. VAD
            logger.info("[3/4] Loading VAD...")
            self._vad = self._init_vad()
            self._vad.on_speech_end(self._on_speech_end)
            self._components_started.append("vad")

            # 4. Audio capture
            logger.info("[4/4] Starting audio capture...")
            self._capture = self._init_capture()
            await self._capture.start()
            self._components_started.append("audio")

            self._running = True
            logger.info("Pipeline started successfully!")

        except Exception as e:
            logger.error(f"Pipeline startup failed: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop all components in reverse order."""
        if not self._components_started:
            return

        logger.info("Stopping pipeline...")
        self._running = False

        # Reverse order shutdown
        for component in reversed(self._components_started):
            try:
                if component == "audio" and self._capture:
                    await self._capture.stop()
                    logger.info("Audio capture stopped")
                elif component == "vad" and self._vad:
                    self._vad.reset()
                    logger.info("VAD reset")
                elif component == "stt" and self._stt:
                    self._stt.unload()
                    logger.info("STT unloaded")
                elif component == "llm" and self._llm:
                    self._llm.unload()
                    logger.info("LLM unloaded")
            except Exception as e:
                logger.error(f"Error stopping {component}: {e}")

        self._components_started.clear()
        logger.info("Pipeline stopped")

    def _on_speech_end(self, audio_segment: np.ndarray) -> None:
        """Callback from VAD when speech ends. Queues audio for STT processing."""
        if self._running and self._capture:
            sample_rate = self._capture.actual_sample_rate
            duration_ms = len(audio_segment) / sample_rate * 1000
            logger.info(f"Speech segment captured: {duration_ms:.0f}ms")

            # Put into async queue (thread-safe via asyncio)
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    self._pending_audio.put_nowait, audio_segment
                )
            except RuntimeError:
                logger.warning("Could not queue audio segment — event loop not running")

    async def _process_audio_loop(self) -> None:
        """Main processing loop: reads audio frames from capture and feeds VAD."""
        if not self._capture or not self._vad:
            return

        frame_size = self._config.get("audio", {}).get("block_size", 512)
        sample_rate = self._capture.actual_sample_rate

        while self._running:
            try:
                # Get latest frame from ring buffer
                frame = self._capture.get_latest_frames(frame_size)
                if len(frame) == frame_size:
                    self._vad.process_frame(frame)
                await asyncio.sleep(0.001)  # ~1ms poll interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.01)

    async def _stt_llm_loop(self) -> None:
        """Process queued audio segments through STT → LLM."""
        while self._running:
            try:
                # Wait for audio segment with timeout
                try:
                    audio_segment = await asyncio.wait_for(
                        self._pending_audio.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_speech(audio_segment)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"STT/LLM processing error: {e}")
                await self.stop()
                raise

    async def _process_speech(self, audio_segment: np.ndarray) -> None:
        """Run STT → LLM on an audio segment.

        In meeting mode: transcribes everything, stores in history,
        and only calls LLM when a question is detected.
        """
        latency = LatencyLog()
        e2e_start = time.perf_counter()

        # Sample rate info
        if self._capture:
            latency.audio_duration_ms = (
                len(audio_segment) / self._capture.actual_sample_rate * 1000
            )

        # Skip very short segments (< 500ms) — likely noise
        if latency.audio_duration_ms < 500:
            logger.debug(f"Segment too short ({latency.audio_duration_ms:.0f}ms), skipping")
            return

        # VAD end time is now
        latency.vad_end_ms = 0.0  # Reference point

        # STT
        stt_start = time.perf_counter()
        text = self._stt.transcribe(audio_segment)
        latency.stt_latency_ms = (time.perf_counter() - stt_start) * 1000

        if not text:
            logger.debug("STT returned empty text, skipping LLM")
            return

        logger.info(f"Transcribed: \"{text}\"")

        # Segment Thai words with spaces for readability
        display_text = self._segment_thai(text)

        # ── Meeting assistant mode ──
        mode = self._config.get("pipeline", {}).get("activation_mode", "push_to_talk")

        if mode == "always_listening":
            # Always store transcript in history
            self._conversation_history.append({"role": "user", "content": text})
            if len(self._conversation_history) > self._max_history:
                self._conversation_history = self._conversation_history[-self._max_history:]

            # Check if this is a question (needs LLM response)
            is_question = self._is_question(text)

            if is_question and not self._is_responding:
                print(f"\n🎤 Q: {display_text}")
                print("🤖 ", end="", flush=True)
                await self._respond_with_history(text)
            else:
                # Just log the transcript — no LLM call
                print(f"  📝 {display_text}")

            return

        # ── Push-to-talk mode (original behavior) ──
        if self._token_callback:
            self._token_callback(f"\n🎤 You: {display_text}\n🤖 ")
        else:
            print(f"\n🎤 You: {display_text}")
            print("🤖 ", end="", flush=True)

        # Check response mode: chat (free text) or extract (Pydantic structured)
        response_mode = self._config.get("pipeline", {}).get("response_mode", "chat")
        full_response = []

        if response_mode == "extract":
            # ── Extract mode: Pydantic-structured output ──
            schema_name = self._config.get("pipeline", {}).get("extract_schema", "PersonInfo")
            schema = get_schema(schema_name)

            llm_start = time.perf_counter()
            try:
                result = self._llm.extract(text, schema)
                latency.llm_ttft_ms = (time.perf_counter() - llm_start) * 1000
                latency.total_e2e_ms = (time.perf_counter() - e2e_start) * 1000

                output = f"✓ {result}"
                full_response = [output]
                if self._token_callback:
                    self._token_callback(output)
                else:
                    print(f"\n{output}")

            except Exception as e:
                latency.llm_ttft_ms = (time.perf_counter() - llm_start) * 1000
                latency.total_e2e_ms = (time.perf_counter() - e2e_start) * 1000
                error_msg = f"✗ Extraction failed: {e}"
                full_response = [error_msg]
                if self._token_callback:
                    self._token_callback(error_msg)
                else:
                    print(f"\n{error_msg}")

        else:
            # ── Chat mode: free-text streaming ──
            request = LLMRequest(user_text=text)

            async for token in self._llm.generate_stream(request):
                if self._token_callback:
                    self._token_callback(token)
                else:
                    print(token, end="", flush=True)
                full_response.append(token)

            if not self._token_callback:
                print()

            latency.llm_ttft_ms = self._llm.last_response.ttft_ms if self._llm.last_response else 0
            latency.total_e2e_ms = (time.perf_counter() - e2e_start) * 1000

        # Log latency
        self._latency_logs.append(latency)
        if self._config.get("pipeline", {}).get("log_latency", True):
            logger.info(
                f"Latency: {latency.to_dict()} | "
                f"Response: \"{''.join(full_response)[:80]}...\""
            )

    @staticmethod
    def _is_question(text: str) -> bool:
        """Detect if the transcribed text is a question.

        Checks for Thai question words and question marks.
        """
        text = text.strip()

        # Question mark
        if text.endswith("?") or text.endswith("？"):
            return True

        # Thai question words
        thai_q_words = ["ไหม", "มั้ย", "อะไร", "ทำไม", "อย่างไร", "ยังไง",
                        "เท่าไหร่", "เท่าไร", "ที่ไหน", "เมื่อไหร่", "เมื่อไร",
                        "ใคร", "กี่", "หรือเปล่า", "ได้ไหม", "ได้มั้ย",
                        "ช่วย", "บอก", "อธิบาย", "สรุป", "สรุปหน่อย"]
        for word in thai_q_words:
            if word in text:
                return True

        # English question words
        en_q_words = ["what", "why", "how", "when", "where", "who", "which",
                       "can you", "could you", "tell me", "explain", "summarize",
                       "do you", "is it", "are there", "will it"]
        text_lower = text.lower()
        for word in en_q_words:
            if word in text_lower:
                return True

        return False

    async def _respond_with_history(self, text: str) -> None:
        """Call LLM with full conversation history as context."""
        self._is_responding = True
        try:
            # Build messages with meeting context
            system = (
                "You are a meeting assistant. You are listening to a live conversation. "
                "You have the full conversation history below. "
                "Answer questions briefly and accurately based on what has been discussed. "
                "If the answer is not in the conversation, say so. "
                "Respond in the same language as the question (Thai or English)."
            )
            messages = [{"role": "system", "content": system}]
            messages.extend(self._conversation_history[-20:])  # last 20 turns for context

            full_response = []
            async for token in self._llm.generate_stream(
                LLMRequest(user_text=text, history=messages)
            ):
                if self._token_callback:
                    self._token_callback(token)
                else:
                    print(token, end="", flush=True)
                full_response.append(token)

            if not self._token_callback:
                print()

            # Store assistant response in history
            response_text = "".join(full_response).strip()
            if response_text:
                self._conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )

            logger.info(
                f"Meeting assistant responded. "
                f"History: {len(self._conversation_history)} turns"
            )
        finally:
            self._is_responding = False

    async def run_push_to_talk(self) -> None:
        """Run in push-to-talk mode.

        Records audio while a key is held, processes on release.
        Uses keyboard input for simplicity (spacebar to talk).
        """
        logger.info("Push-to-talk mode: press ENTER to start recording, ENTER again to stop")

        while self._running:
            try:
                # Wait for user to press Enter to start recording
                await asyncio.get_event_loop().run_in_executor(None, input)
                if not self._running:
                    break

                logger.info("🎤 Recording... (press ENTER to stop)")
                if self._capture:
                    self._capture.ring_buffer.clear()

                # Wait for Enter to stop recording
                await asyncio.get_event_loop().run_in_executor(None, input)
                if not self._running:
                    break

                # Get recorded audio
                if self._capture:
                    audio = self._capture.ring_buffer.read_all()
                    if len(audio) > 0:
                        await self._process_speech(audio)

            except asyncio.CancelledError:
                break
            except EOFError:
                break

    async def run_always_listening(self) -> None:
        """Run in always-listening mode with VAD-driven speech detection."""
        logger.info("Always-listening mode: speak naturally, VAD will detect boundaries")

        # Run audio processing and STT/LLM loops concurrently
        await asyncio.gather(
            self._process_audio_loop(),
            self._stt_llm_loop(),
        )

    async def run(self) -> None:
        """Run the pipeline based on configured activation mode."""
        mode = self._config.get("pipeline", {}).get("activation_mode", "push_to_talk")

        try:
            if mode == "push_to_talk":
                await self.run_push_to_talk()
            elif mode == "always_listening":
                await self.run_always_listening()
            else:
                raise ValueError(f"Unknown activation mode: {mode}")
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    def get_latency_stats(self) -> dict:
        """Get aggregate latency statistics."""
        if not self._latency_logs:
            return {"requests": 0}

        ttfts = [l.llm_ttft_ms for l in self._latency_logs]
        e2es = [l.total_e2e_ms for l in self._latency_logs]
        stts = [l.stt_latency_ms for l in self._latency_logs]

        def percentile(data: List[float], p: float) -> float:
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "requests": len(self._latency_logs),
            "ttft_ms": {
                "p50": round(percentile(ttfts, 50), 1),
                "p95": round(percentile(ttfts, 95), 1),
                "avg": round(sum(ttfts) / len(ttfts), 1),
            },
            "stt_ms": {
                "p50": round(percentile(stts, 50), 1),
                "avg": round(sum(stts) / len(stts), 1),
            },
            "e2e_ms": {
                "p50": round(percentile(e2es, 50), 1),
                "p95": round(percentile(e2es, 95), 1),
                "avg": round(sum(e2es) / len(e2es), 1),
            },
        }
