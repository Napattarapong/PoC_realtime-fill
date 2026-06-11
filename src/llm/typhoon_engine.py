"""Typhoon LLM engine — GGUF model via llama-cpp-python with Metal.

Two modes:
1. Chat mode — free-text assistant responses
2. Extract mode — Pydantic-structured output (JSON) from user speech

The extract mode prompts the LLM to return JSON matching a Pydantic schema,
then validates the output. Only the requested fields come through — no rambling.
"""

import json
import logging
import os
import re
import time
from typing import AsyncGenerator, Callable, Dict, List, Optional, Type, TypeVar

import psutil
from pydantic import BaseModel, Field, ValidationError, validator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ── Pydantic Models ──────────────────────────────────────────────

class LLMRequest(BaseModel):
    """Structured input to the LLM."""
    user_text: str = Field(..., min_length=1, description="User's transcribed speech")
    history: Optional[List[Dict[str, str]]] = Field(default=None)

    @validator('user_text')
    def clean_text(cls, v):
        return v.strip()


class LLMResponse(BaseModel):
    """Metadata about an LLM call."""
    raw_text: str = Field(..., description="Raw generated text")
    ttft_ms: float = Field(..., description="Time to first token")
    total_ms: float = Field(..., description="Total generation time")
    tokens_generated: int = Field(..., description="Token count")


class PersonInfo(BaseModel):
    """Extract person info from speech."""
    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age in years")


class GenerationParams(BaseModel):
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    max_tokens: int = Field(default=64, ge=1, le=512)
    repeat_penalty: float = Field(default=1.2, ge=1.0, le=2.0)

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "GenerationParams":
        return cls(**(d or {}))


# ── Schema Registry ──────────────────────────────────────────────

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "PersonInfo": PersonInfo,
}


def get_schema(name: str) -> Type[BaseModel]:
    """Look up a Pydantic schema by name."""
    if name not in SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown schema '{name}'. Available: {list(SCHEMA_REGISTRY.keys())}"
        )
    return SCHEMA_REGISTRY[name]


# ── JSON Extraction Helpers ───────────────────────────────────────

def _build_extract_prompt(user_text: str, schema: Type[T]) -> str:
    """Build a prompt that forces JSON output matching a Pydantic schema.

    Tells the LLM exactly what fields to extract and that it MUST return
    only valid JSON — no extra text, no conversation.
    """
    schema_json = schema.schema()
    properties = schema_json.get("properties", {})
    fields_desc = ", ".join(
        f'"{k}": {v.get("description", v.get("type", "string"))}'
        for k, v in properties.items()
    )

    return (
        f"Extract information from the user's text and return ONLY valid JSON.\n"
        f"Schema: {{{fields_desc}}}\n"
        f"Rules: Return ONLY the JSON object. No other text.\n"
        f'Example: {{"name": "John", "age": 25}}\n'
        f"\nUser: {user_text}\n"
        f"JSON:"
    )


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON object from LLM output, even if wrapped in markdown."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } in the text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


# ── Engine ────────────────────────────────────────────────────────

class TyphoonEngine:
    """LLM inference engine with chat and structured extraction modes."""

    STOP_TOKENS = ["User:", "\n\n", "\n###"]

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 1024,
        system_prompt: str = "",
        generation_params: Optional[dict] = None,
    ):
        self._model_path = os.path.expanduser(model_path)
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._system_prompt = system_prompt
        self._gen_params = GenerationParams.from_dict(generation_params)

        self._llm = None
        self._last_response: Optional[LLMResponse] = None

    @property
    def last_response(self) -> Optional[LLMResponse]:
        return self._last_response

    @property
    def last_ttft_ms(self) -> float:
        return self._last_response.ttft_ms if self._last_response else 0.0

    @property
    def memory_rss_gb(self) -> float:
        return psutil.Process().memory_info().rss / (1024 ** 3)

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """Load the Typhoon GGUF model into memory with Metal acceleration."""
        from llama_cpp import Llama

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Typhoon model not found at: {self._model_path}\n"
                f"Run ./scripts/setup_models.sh to download and convert the model."
            )

        logger.info(
            f"Loading Typhoon model: {self._model_path} "
            f"(n_gpu_layers={self._n_gpu_layers}, n_ctx={self._n_ctx})"
        )

        self._llm = Llama(
            model_path=self._model_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            verbose=False,
        )

        logger.info(f"Typhoon model loaded. Memory RSS: {self.memory_rss_gb:.2f} GB")

    def warmup_prompt_cache(self) -> None:
        """Warmup — one dummy completion so the model is ready."""
        if not self._llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info("Warming up model...")
        t0 = time.perf_counter()

        prompt = self._build_chat_prompt("hi")
        self._llm.create_completion(
            prompt, max_tokens=1, temperature=0.0, stop=self.STOP_TOKENS
        )

        logger.info(f"Model warmup complete in {(time.perf_counter()-t0)*1000:.0f}ms")

    # ── Chat Mode ─────────────────────────────────────────────────

    def _build_chat_prompt(self, user_text: str, history: Optional[List[Dict]] = None) -> str:
        """Build a simple text-to-text prompt."""
        parts = []
        if self._system_prompt:
            parts.append(self._system_prompt)
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"\n{role.capitalize()}: {content}")
        parts.append(f"\nUser: {user_text}")
        parts.append("\nAssistant:")
        return "".join(parts)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a free-text response. Returns LLMResponse."""
        if not self._llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = self._build_chat_prompt(request.user_text, request.history)
        p = self._gen_params

        t0 = time.perf_counter()
        result = self._llm.create_completion(
            prompt, max_tokens=p.max_tokens, temperature=p.temperature,
            top_p=p.top_p, top_k=p.top_k, repeat_penalty=p.repeat_penalty,
            stop=self.STOP_TOKENS, stream=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        raw_text = result["choices"][0]["text"].strip()
        self._last_response = LLMResponse(
            raw_text=raw_text, ttft_ms=total_ms, total_ms=total_ms,
            tokens_generated=result["usage"]["completion_tokens"],
        )
        logger.info(f"LLM chat: TTFT={total_ms:.0f}ms, tokens={self._last_response.tokens_generated}")
        return self._last_response

    async def generate_stream(
        self,
        request: LLMRequest,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream free-text response. Yields tokens."""
        if not self._llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = self._build_chat_prompt(request.user_text, request.history)
        p = self._gen_params

        t0 = time.perf_counter()
        first_token = True
        tokens = []
        collected = []

        stream = self._llm.create_completion(
            prompt, max_tokens=p.max_tokens, temperature=p.temperature,
            top_p=p.top_p, top_k=p.top_k, repeat_penalty=p.repeat_penalty,
            stop=self.STOP_TOKENS, stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("text", "")
            if not delta:
                continue
            if first_token:
                ttft_ms = (time.perf_counter() - t0) * 1000
                first_token = False
                logger.info(f"LLM TTFT: {ttft_ms:.0f}ms")
            tokens.append(delta)
            collected.append(delta)
            if token_callback:
                token_callback(delta)
            yield delta
            await _async_yield()

        total_ms = (time.perf_counter() - t0) * 1000
        self._last_response = LLMResponse(
            raw_text="".join(collected).strip(),
            ttft_ms=ttft_ms if not first_token else total_ms,
            total_ms=total_ms, tokens_generated=len(tokens),
        )

    # ── Extract Mode (Pydantic-structured output) ─────────────────

    def extract(self, user_text: str, schema: Type[T]) -> T:
        """Extract structured data from user speech using Pydantic schema.

        Prompts the LLM to return JSON, then validates with Pydantic.
        Only the fields defined in the schema are extracted — no rambling.

        Args:
            user_text: Transcribed speech (e.g. "My name is John, I'm 25")
            schema: Pydantic model class (e.g. PersonInfo)

        Returns:
            Validated Pydantic instance (e.g. PersonInfo(name="John", age=25))

        Raises:
            ValueError: If LLM output can't be parsed or validated
        """
        if not self._llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = _build_extract_prompt(user_text, schema)
        p = self._gen_params

        t0 = time.perf_counter()
        result = self._llm.create_completion(
            prompt, max_tokens=p.max_tokens, temperature=0.0,
            stop=["\n", "User:"], stream=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        raw_text = result["choices"][0]["text"].strip()
        self._last_response = LLMResponse(
            raw_text=raw_text, ttft_ms=total_ms, total_ms=total_ms,
            tokens_generated=result["usage"]["completion_tokens"],
        )
        logger.info(f"LLM extract: {total_ms:.0f}ms, raw='{raw_text}'")

        # Parse JSON and validate with Pydantic
        data = _parse_json_from_text(raw_text)

        try:
            return schema(**data)
        except ValidationError as e:
            raise ValueError(
                f"LLM output failed Pydantic validation for {schema.__name__}: {e}\n"
                f"Raw output: {raw_text}"
            )

    def unload(self) -> None:
        """Release model resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._last_response = None
            logger.info("Typhoon model unloaded")


async def _async_yield():
    import asyncio
    await asyncio.sleep(0)
