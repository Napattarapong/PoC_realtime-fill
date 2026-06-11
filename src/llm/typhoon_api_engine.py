"""Typhoon LLM engine — cloud API via OpenAI-compatible endpoint.

Uses the Typhoon API (api.opentyphoon.ai) for fast inference.
Drop-in replacement for the local GGUF engine — same interface.

Two modes:
1. Chat mode — free-text assistant responses (streaming)
2. Extract mode — Pydantic-structured output (JSON) from user speech
"""

import json
import logging
import os
import re
import time
from typing import AsyncGenerator, Callable, Dict, List, Optional, Type, TypeVar

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
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1, le=8192)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)

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
    """Build a prompt that forces JSON output matching a Pydantic schema."""
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
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]}")


# ── Engine ────────────────────────────────────────────────────────

class TyphoonEngine:
    """LLM engine using Typhoon cloud API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "typhoon-v2.5-30b-a3b-instruct",
        base_url: str = "https://api.opentyphoon.ai/v1",
        system_prompt: str = "",
        generation_params: Optional[dict] = None,
        # Legacy params (ignored, kept for config compatibility)
        model_path: str = "",
        n_gpu_layers: int = -1,
        n_ctx: int = 1024,
    ):
        self._api_key = api_key or os.environ.get("TYPHOON_API_KEY", "")
        if not self._api_key:
            # Try loading from .env file
            env_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TYPHOON_API_KEY="):
                            self._api_key = line.split("=", 1)[1].strip()
                            break
        self._model = model
        self._base_url = base_url
        self._system_prompt = system_prompt
        self._gen_params = GenerationParams.from_dict(generation_params)

        self._client = None
        self._last_response: Optional[LLMResponse] = None

    @property
    def last_response(self) -> Optional[LLMResponse]:
        return self._last_response

    @property
    def last_ttft_ms(self) -> float:
        return self._last_response.ttft_ms if self._last_response else 0.0

    @property
    def memory_rss_gb(self) -> float:
        return 0.0  # Cloud API — no local memory

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def load(self) -> None:
        """Initialize the OpenAI client for Typhoon API."""
        from openai import OpenAI

        if not self._api_key:
            raise ValueError(
                "Typhoon API key required. Set TYPHOON_API_KEY env var "
                "or add api_key to config."
            )

        logger.info(
            f"Connecting to Typhoon API: model={self._model}"
        )

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        logger.info("Typhoon API client ready")

    def warmup_prompt_cache(self) -> None:
        """Quick warmup request to verify API connectivity."""
        if not self._client:
            raise RuntimeError("API client not initialized. Call load() first.")

        logger.info("Warming up API connection...")
        t0 = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"API warmup complete in {elapsed:.0f}ms")

    # ── Chat Mode ─────────────────────────────────────────────────

    def _build_messages(
        self, user_text: str, history: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """Build OpenAI-style messages array.

        If history is provided and already contains a system prompt as the
        first message (meeting assistant mode), use it directly.
        Otherwise, build from system_prompt + history + user text.
        """
        # Meeting assistant mode: history already has system + full context
        if history and history[0].get("role") == "system":
            messages = list(history)
            # Add user text only if it's not already the last message
            if not (messages and messages[-1].get("role") == "user"
                    and messages[-1].get("content") == user_text):
                messages.append({"role": "user", "content": user_text})
            return messages

        # Standard mode: build from scratch
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a free-text response. Returns LLMResponse."""
        if not self._client:
            raise RuntimeError("API client not initialized. Call load() first.")

        messages = self._build_messages(request.user_text, request.history)
        p = self._gen_params

        t0 = time.perf_counter()
        result = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=p.max_tokens,
            temperature=p.temperature,
            top_p=p.top_p,
            extra_body={"repetition_penalty": p.repetition_penalty},
            stream=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        raw_text = result.choices[0].message.content.strip()
        self._last_response = LLMResponse(
            raw_text=raw_text, ttft_ms=total_ms, total_ms=total_ms,
            tokens_generated=result.usage.completion_tokens,
        )
        logger.info(f"LLM chat: TTFT={total_ms:.0f}ms, tokens={self._last_response.tokens_generated}")
        return self._last_response

    async def generate_stream(
        self,
        request: LLMRequest,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream free-text response. Yields tokens."""
        if not self._client:
            raise RuntimeError("API client not initialized. Call load() first.")

        messages = self._build_messages(request.user_text, request.history)
        p = self._gen_params

        t0 = time.perf_counter()
        first_token = True
        tokens = []
        collected = []

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=p.max_tokens,
            temperature=p.temperature,
            top_p=p.top_p,
            extra_body={"repetition_penalty": p.repetition_penalty},
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
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
        """Extract structured data from user speech using Pydantic schema."""
        if not self._client:
            raise RuntimeError("API client not initialized. Call load() first.")

        prompt = _build_extract_prompt(user_text, schema)

        t0 = time.perf_counter()
        result = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._gen_params.max_tokens,
            temperature=0.0,
            stream=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        raw_text = result.choices[0].message.content.strip()
        self._last_response = LLMResponse(
            raw_text=raw_text, ttft_ms=total_ms, total_ms=total_ms,
            tokens_generated=result.usage.completion_tokens,
        )
        logger.info(f"LLM extract: {total_ms:.0f}ms, raw='{raw_text}'")

        data = _parse_json_from_text(raw_text)

        try:
            return schema(**data)
        except ValidationError as e:
            raise ValueError(
                f"LLM output failed Pydantic validation for {schema.__name__}: {e}\n"
                f"Raw output: {raw_text}"
            )

    def unload(self) -> None:
        """Release API client."""
        self._client = None
        self._last_response = None
        logger.info("Typhoon API client closed")


async def _async_yield():
    import asyncio
    await asyncio.sleep(0)
