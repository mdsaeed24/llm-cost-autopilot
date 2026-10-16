"""Shared fixtures. No test ever talks to a real LLM: every SDK client is a fake."""

from __future__ import annotations

import hashlib
import importlib.util
import random
import re
from collections.abc import Iterator, Sequence
from collections.abc import Callable
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import httpx2
import numpy as np
import openai
import pytest
from openai.types.chat import ChatCompletion

import autopilot.classifier.embeddings as embeddings_module
from autopilot.llm.client import LLMClient
from autopilot.llm.schemas import ModelConfig, ModelPricing, ThinkingConfig
from autopilot.settings import PROJECT_ROOT, ClientSettings

_REQUEST = httpx2.Request("POST", "http://test.local/v1/chat/completions")


class HashingEmbedder:
    """Offline stand-in for Ollama embeddings: bag of hashed words → unit vector. Deterministic."""

    dim = 64

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str], *, model: str) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture(autouse=True, scope="session")
def _fake_embedder() -> Iterator[HashingEmbedder]:
    """No test ever calls the real Ollama embedding model."""
    fake = HashingEmbedder()
    embeddings_module.set_embedder(fake)
    yield fake
    embeddings_module.set_embedder(None)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real .env out of tests, and default to no API key."""
    monkeypatch.setattr("autopilot.settings.load_env", lambda: None)
    for name in ("DEEPSEEK_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "DEEPSEEK_PRO_THINKING", "AUTOPILOT_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)


# ---------- fake completions and errors ----------

def make_completion(
    content: str = "ok",
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    deepseek_cache_hit: int | None = None,
    openai_cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    include_usage: bool = True,
) -> ChatCompletion:
    """Build a realistic ChatCompletion, optionally with provider-specific usage fields."""
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if deepseek_cache_hit is not None:
        usage["prompt_cache_hit_tokens"] = deepseek_cache_hit
        usage["prompt_cache_miss_tokens"] = prompt_tokens - deepseek_cache_hit
    if openai_cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": openai_cached_tokens}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
            ],
            "usage": usage if include_usage else None,
        }
    )


def rate_limit_error(retry_after: float | None = None) -> openai.RateLimitError:
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return openai.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=_REQUEST, headers=headers), body=None
    )


def server_error() -> openai.InternalServerError:
    return openai.InternalServerError("boom", response=httpx2.Response(503, request=_REQUEST), body=None)


def bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError("bad request", response=httpx2.Response(400, request=_REQUEST), body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


# ---------- fake SDK client ----------

Outcome = ChatCompletion | Exception | Callable[[dict[str, Any]], Any]


class FakeCompletions:
    """Stands in for ``client.chat.completions``. Plays back outcomes in order; the last one repeats."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if callable(outcome) and not isinstance(outcome, ChatCompletion):
            outcome = await outcome(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAI:
    """Minimal AsyncOpenAI replacement exposing ``chat.completions.create`` and ``close``."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeSleep:
    """Records backoff delays instead of actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def make_client(fake: FakeOpenAI, **settings: Any) -> tuple[LLMClient, FakeSleep]:
    """An LLMClient whose every provider is backed by ``fake``."""
    sleep = FakeSleep()
    client = LLMClient(
        ClientSettings(**settings),
        client_factory=lambda model, _settings: (model.resolve_api_key(), fake)[1],
        sleep=sleep,
        rng=random.Random(0),
    )
    return client, sleep


# ---------- model fixtures ----------

@pytest.fixture
def ollama_model() -> ModelConfig:
    return ModelConfig(
        name="local-llama",
        provider="ollama",
        model_id="llama3.2:3b",
        base_url="http://localhost:11434/v1",
        quality_tier="low",
        typical_latency_ms=4000,
        pricing=ModelPricing(input_cache_miss_per_1m=0, input_cache_hit_per_1m=0, output_per_1m=0),
    )


@pytest.fixture
def flash_model() -> ModelConfig:
    return ModelConfig(
        name="deepseek-flash",
        provider="deepseek",
        model_id="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        quality_tier="medium",
        typical_latency_ms=3000,
        pricing=ModelPricing(input_cache_miss_per_1m=0.30, input_cache_hit_per_1m=0.006, output_per_1m=1.20),
        thinking=ThinkingConfig(enabled=False),
    )


@pytest.fixture
def pro_thinking_model(flash_model: ModelConfig) -> ModelConfig:
    return flash_model.model_copy(
        update={
            "name": "deepseek-v4-pro",
            "model_id": "deepseek-v4-pro",
            "quality_tier": "high",
            "pricing": ModelPricing(input_cache_miss_per_1m=1.32, input_cache_hit_per_1m=0.044, output_per_1m=3.96),
            "thinking": ThinkingConfig(enabled=True, reasoning_effort="max"),
        }
    )


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    return "sk-test"


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module so tests can call its ``main``."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
