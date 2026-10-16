"""One async entry point, :func:`send_request`, for every model tier.

All providers speak the OpenAI chat-completions protocol, so a single SDK with
different base URLs covers Ollama and DeepSeek. This module adds what the raw
SDK lacks: per-provider concurrency limits, retries with exponential backoff,
cost calculation, and a provider-independent :class:`LLMResponse`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from autopilot.llm.pricing import compute_cost
from autopilot.llm.schemas import ChatMessage, LLMResponse, ModelConfig
from autopilot.settings import ClientSettings, load_settings

logger = logging.getLogger(__name__)

# Transient failures worth retrying. APITimeoutError subclasses APIConnectionError.
# 4xx errors (bad request, auth, not found) will fail the same way again, so they are not retried.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

ClientFactory = Callable[[ModelConfig, ClientSettings], AsyncOpenAI]
Sleeper = Callable[[float], Awaitable[None]]
MessageLike = ChatMessage | dict[str, str]


class LLMRequestError(RuntimeError):
    """A request failed permanently (non-retryable error, or retries exhausted)."""

    def __init__(self, model_name: str, attempts: int, cause: Exception) -> None:
        super().__init__(f"{model_name}: failed after {attempts} attempt(s): {type(cause).__name__}: {cause}")
        self.model_name = model_name
        self.attempts = attempts
        self.cause = cause


def _default_client_factory(model: ModelConfig, settings: ClientSettings) -> AsyncOpenAI:
    # max_retries=0: we run our own retry loop so every attempt is counted and logged.
    return AsyncOpenAI(
        api_key=model.resolve_api_key(),
        base_url=model.base_url,
        timeout=settings.request_timeout_s,
        max_retries=0,
    )


def build_request_kwargs(
    messages: Sequence[ChatMessage],
    model: ModelConfig,
    *,
    max_tokens: int,
    temperature: float | None = None,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Build the ``chat.completions.create`` arguments, including DeepSeek's thinking toggle."""
    kwargs: dict[str, Any] = {
        "model": model.model_id,
        "messages": [m.model_dump() for m in messages],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        # Both DeepSeek and Ollama support this; the prompt must also ask for JSON.
        kwargs["response_format"] = {"type": "json_object"}
    if model.thinking is not None:
        # DeepSeek defaults thinking to ON, so "disabled" must be sent explicitly.
        mode = "enabled" if model.thinking.enabled else "disabled"
        kwargs["extra_body"] = {"thinking": {"type": mode}}
        if model.thinking.enabled:
            kwargs["reasoning_effort"] = model.thinking.reasoning_effort
    return kwargs


def backoff_delay(
    attempt: int,
    *,
    base_s: float,
    max_s: float,
    rng: random.Random,
    retry_after_s: float | None = None,
) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based).

    Exponential backoff with "equal jitter": wait half the exponential delay plus a random
    share of the other half, so many clients that failed together don't retry together.
    A server-provided ``Retry-After`` takes priority.
    """
    if retry_after_s is not None and retry_after_s > 0:
        return min(max_s, retry_after_s)
    exponential = min(max_s, base_s * 2 ** (attempt - 1))
    return exponential / 2 + rng.uniform(0, exponential / 2)


def _retry_after_seconds(error: Exception) -> float | None:
    if not isinstance(error, openai.APIStatusError):
        return None
    raw = error.response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff


def _usage_counts(completion: ChatCompletion) -> tuple[int, int, int, int]:
    """Return (prompt_tokens, cache_hit_tokens, completion_tokens, reasoning_tokens).

    DeepSeek reports ``prompt_cache_hit_tokens``; OpenAI-style servers (including Ollama)
    report ``prompt_tokens_details.cached_tokens``. Either way we get the cache-hit count.
    """
    usage = completion.usage
    if usage is None:
        logger.warning("Response from %s had no usage block; cost recorded as 0", completion.model)
        return 0, 0, 0, 0
    prompt = usage.prompt_tokens or 0
    extra = usage.model_extra or {}
    cache_hit = extra.get("prompt_cache_hit_tokens")
    if cache_hit is None:
        details = usage.prompt_tokens_details
        cache_hit = (details.cached_tokens or 0) if details else 0
    reasoning_details = usage.completion_tokens_details
    reasoning = (reasoning_details.reasoning_tokens or 0) if reasoning_details else 0
    return prompt, min(int(cache_hit), prompt), usage.completion_tokens or 0, reasoning


def to_llm_response(
    completion: ChatCompletion,
    model: ModelConfig,
    *,
    latency_ms: float,
    attempts: int,
) -> LLMResponse:
    """Convert a raw SDK completion into our provider-independent :class:`LLMResponse`."""
    prompt, cache_hit, output, reasoning = _usage_counts(completion)
    choice = completion.choices[0] if completion.choices else None
    text = (choice.message.content or "") if choice else ""
    finish_reason = (choice.finish_reason if choice else None) or "unknown"
    cost = compute_cost(
        model.pricing,
        cache_miss_tokens=prompt - cache_hit,
        cache_hit_tokens=cache_hit,
        output_tokens=output,
    )
    return LLMResponse(
        model_name=model.name,
        model_id=model.model_id,
        provider=model.provider,
        output_text=text,
        input_tokens=prompt,
        cached_input_tokens=cache_hit,
        output_tokens=output,
        reasoning_tokens=reasoning,
        latency_ms=round(latency_ms, 1),
        cost_usd=cost,
        finish_reason=finish_reason,
        attempts=attempts,
    )


class LLMClient:
    """Sends chat requests to any configured model with concurrency limits and retries.

    Create one per process (event loop). Semaphores and HTTP clients are created lazily
    and reused, so connection pools stay warm.
    """

    def __init__(
        self,
        settings: ClientSettings | None = None,
        *,
        client_factory: ClientFactory = _default_client_factory,
        sleep: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings or load_settings().client
        self._client_factory = client_factory
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clients: dict[tuple[str, str | None], AsyncOpenAI] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _client_for(self, model: ModelConfig) -> AsyncOpenAI:
        key = (model.base_url, model.api_key_env)
        if key not in self._clients:
            self._clients[key] = self._client_factory(model, self.settings)
        return self._clients[key]

    def _semaphore_for(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._semaphores:
            self._semaphores[provider] = asyncio.Semaphore(self.settings.concurrency_for(provider))
        return self._semaphores[provider]

    async def send(
        self,
        messages: Sequence[MessageLike],
        model: ModelConfig,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send one chat request and return a normalised, priced response.

        Raises:
            MissingAPIKeyError: the model's API key is not configured.
            LLMRequestError: the request failed permanently.
        """
        chat = [m if isinstance(m, ChatMessage) else ChatMessage.model_validate(m) for m in messages]
        kwargs = build_request_kwargs(
            chat,
            model,
            max_tokens=max_tokens or self.settings.max_output_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        client = self._client_for(model)

        # Latency is measured after we get a concurrency slot: time spent queueing
        # in our own semaphore is a client-side artefact, not model latency.
        async with self._semaphore_for(model.provider):
            started = time.perf_counter()
            completion, attempts = await self._create_with_retries(client, kwargs, model)
            latency_ms = (time.perf_counter() - started) * 1000

        return to_llm_response(completion, model, latency_ms=latency_ms, attempts=attempts)

    async def _create_with_retries(
        self, client: AsyncOpenAI, kwargs: dict[str, Any], model: ModelConfig
    ) -> tuple[ChatCompletion, int]:
        max_attempts = self.settings.max_attempts
        for attempt in range(1, max_attempts + 1):
            try:
                completion = await client.chat.completions.create(**kwargs)
                return completion, attempt
            except RETRYABLE_ERRORS as error:
                if attempt == max_attempts:
                    raise LLMRequestError(model.name, attempt, error) from error
                delay = backoff_delay(
                    attempt,
                    base_s=self.settings.backoff_base_s,
                    max_s=self.settings.backoff_max_s,
                    rng=self._rng,
                    retry_after_s=_retry_after_seconds(error),
                )
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    model.name, attempt, max_attempts, type(error).__name__, delay,
                )
                await self._sleep(delay)
            except openai.OpenAIError as error:
                raise LLMRequestError(model.name, attempt, error) from error
        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        """Close all underlying HTTP clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


_default_client: LLMClient | None = None


def get_default_client() -> LLMClient:
    """Process-wide client built from ``config/settings.yaml``."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


async def send_request(
    messages: Sequence[MessageLike],
    model_config: ModelConfig,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> LLMResponse:
    """Send ``messages`` to ``model_config`` and return an :class:`LLMResponse`.

    This is the single entry point the rest of the system uses for every tier.
    """
    return await get_default_client().send(
        messages, model_config, max_tokens=max_tokens, temperature=temperature
    )
