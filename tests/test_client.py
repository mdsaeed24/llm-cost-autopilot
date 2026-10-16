"""LLM client: request building, response parsing, retries, concurrency. All calls are faked."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest
from pydantic import ValidationError

import autopilot.llm.client as client_module
from autopilot.llm.client import LLMClient, LLMRequestError, backoff_delay, build_request_kwargs, send_request
from autopilot.llm.schemas import ChatMessage, MissingAPIKeyError, ModelConfig
from autopilot.settings import ClientSettings
from tests.conftest import (
    FakeOpenAI,
    bad_request_error,
    connection_error,
    make_client,
    make_completion,
    rate_limit_error,
    server_error,
)

HELLO = [ChatMessage(role="user", content="hello")]


# ---------- request building ----------

def test_deepseek_thinking_disabled_is_sent_explicitly(flash_model: ModelConfig) -> None:
    kwargs = build_request_kwargs(HELLO, flash_model, max_tokens=64)
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in kwargs
    assert kwargs["max_tokens"] == 64
    assert kwargs["model"] == "deepseek-flash"


def test_deepseek_thinking_enabled_sends_effort(pro_thinking_model: ModelConfig) -> None:
    kwargs = build_request_kwargs(HELLO, pro_thinking_model, max_tokens=64)
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kwargs["reasoning_effort"] == "max"


def test_ollama_gets_no_thinking_params(ollama_model: ModelConfig) -> None:
    kwargs = build_request_kwargs(HELLO, ollama_model, max_tokens=64, temperature=0.2)
    assert "extra_body" not in kwargs and "reasoning_effort" not in kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


# ---------- response parsing ----------

async def test_deepseek_usage_is_priced_with_cache_split(flash_model: ModelConfig, api_key: str) -> None:
    fake = FakeOpenAI([make_completion("hi", prompt_tokens=1000, completion_tokens=2000, deepseek_cache_hit=400)])
    client, _ = make_client(fake)
    response = await client.send(HELLO, flash_model)
    assert response.output_text == "hi"
    assert (response.input_tokens, response.cached_input_tokens, response.output_tokens) == (1000, 400, 2000)
    assert response.cost_usd == pytest.approx((600 * 0.30 + 400 * 0.006 + 2000 * 1.20) / 1e6)
    assert response.model_name == "deepseek-flash" and response.provider == "deepseek"
    assert response.finish_reason == "stop" and not response.truncated
    assert response.attempts == 1
    assert response.latency_ms >= 0


async def test_openai_style_cached_tokens_and_reasoning(pro_thinking_model: ModelConfig, api_key: str) -> None:
    fake = FakeOpenAI([make_completion(prompt_tokens=100, openai_cached_tokens=30, reasoning_tokens=80)])
    client, _ = make_client(fake)
    response = await client.send(HELLO, pro_thinking_model)
    assert response.cached_input_tokens == 30
    assert response.reasoning_tokens == 80


async def test_length_finish_reason_marks_truncated(ollama_model: ModelConfig) -> None:
    client, _ = make_client(FakeOpenAI([make_completion(finish_reason="length")]))
    response = await client.send(HELLO, ollama_model)
    assert response.truncated
    assert response.model_dump()["truncated"] is True  # serialised for logs/reports
    assert response.cost_usd == 0


async def test_missing_usage_costs_zero(flash_model: ModelConfig, api_key: str) -> None:
    client, _ = make_client(FakeOpenAI([make_completion(include_usage=False)]))
    response = await client.send(HELLO, flash_model)
    assert response.cost_usd == 0 and response.input_tokens == 0


async def test_default_max_tokens_comes_from_settings(ollama_model: ModelConfig) -> None:
    fake = FakeOpenAI([make_completion()])
    client, _ = make_client(fake, max_output_tokens=321)
    await client.send(HELLO, ollama_model)
    assert fake.completions.calls[0]["max_tokens"] == 321


async def test_dict_messages_are_validated(ollama_model: ModelConfig) -> None:
    client, _ = make_client(FakeOpenAI([make_completion()]))
    await client.send([{"role": "user", "content": "hi"}], ollama_model)
    with pytest.raises(ValidationError):
        await client.send([{"role": "robot", "content": "hi"}], ollama_model)


# ---------- retries ----------

async def test_rate_limit_retries_and_honours_retry_after(flash_model: ModelConfig, api_key: str) -> None:
    fake = FakeOpenAI([rate_limit_error(retry_after=2), make_completion()])
    client, sleep = make_client(fake)
    response = await client.send(HELLO, flash_model)
    assert response.attempts == 2
    assert sleep.delays == [2.0]
    assert len(fake.completions.calls) == 2


async def test_transient_errors_back_off_exponentially(ollama_model: ModelConfig) -> None:
    fake = FakeOpenAI([connection_error(), server_error(), connection_error(), make_completion()])
    client, sleep = make_client(fake, max_attempts=4, backoff_base_s=1.0)
    response = await client.send(HELLO, ollama_model)
    assert response.attempts == 4
    # equal jitter: attempt n waits between half and all of base * 2^(n-1)
    for attempt, delay in enumerate(sleep.delays, start=1):
        assert 2 ** (attempt - 1) / 2 <= delay <= 2 ** (attempt - 1)


async def test_retries_exhausted_raises(ollama_model: ModelConfig) -> None:
    fake = FakeOpenAI([server_error()])
    client, sleep = make_client(fake, max_attempts=3)
    with pytest.raises(LLMRequestError) as info:
        await client.send(HELLO, ollama_model)
    assert info.value.attempts == 3
    assert len(fake.completions.calls) == 3
    assert len(sleep.delays) == 2


async def test_client_errors_are_not_retried(ollama_model: ModelConfig) -> None:
    fake = FakeOpenAI([bad_request_error()])
    client, sleep = make_client(fake)
    with pytest.raises(LLMRequestError, match="BadRequestError") as info:
        await client.send(HELLO, ollama_model)
    assert info.value.attempts == 1
    assert sleep.delays == []


def test_backoff_delay_caps_and_retry_after() -> None:
    rng = random.Random(0)
    assert backoff_delay(10, base_s=1, max_s=30, rng=rng) <= 30
    assert backoff_delay(1, base_s=1, max_s=30, rng=rng, retry_after_s=100) == 30
    assert 2 <= backoff_delay(3, base_s=1, max_s=30, rng=rng) <= 4


# ---------- keys, concurrency, client reuse ----------

async def test_missing_api_key_fails_before_any_call(flash_model: ModelConfig) -> None:
    fake = FakeOpenAI([make_completion()])
    client, _ = make_client(fake)
    with pytest.raises(MissingAPIKeyError):
        await client.send(HELLO, flash_model)
    assert fake.completions.calls == []


async def test_concurrency_limit_is_per_provider(ollama_model: ModelConfig) -> None:
    in_flight = 0
    peak = 0

    async def slow(_kwargs: dict[str, Any]) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return make_completion()

    client, _ = make_client(FakeOpenAI([slow]), concurrency_default=5, concurrency_per_provider={"ollama": 2})
    await asyncio.gather(*(client.send(HELLO, ollama_model) for _ in range(10)))
    assert peak == 2


async def test_sdk_client_is_reused_and_closed(ollama_model: ModelConfig) -> None:
    fake = FakeOpenAI([make_completion()])
    created = 0

    def factory(model: ModelConfig, settings: ClientSettings) -> FakeOpenAI:
        nonlocal created
        created += 1
        return fake

    client = LLMClient(ClientSettings(), client_factory=factory)  # type: ignore[arg-type]
    await client.send(HELLO, ollama_model)
    await client.send(HELLO, ollama_model)
    assert created == 1
    await client.aclose()
    assert fake.closed


async def test_module_level_send_request(ollama_model: ModelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = make_client(FakeOpenAI([make_completion("from default client")]))
    monkeypatch.setattr(client_module, "_default_client", client)
    response = await send_request(HELLO, ollama_model)
    assert response.output_text == "from default client"
