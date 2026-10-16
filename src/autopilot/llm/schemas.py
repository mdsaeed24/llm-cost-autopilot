"""Typed data structures shared by every provider: model configs, messages, responses."""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    field_validator,
)

Provider = Literal["ollama", "deepseek"]
QualityTier = Literal["low", "medium", "high"]
Role = Literal["system", "user", "assistant"]
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# Ollama ignores the API key, but the OpenAI SDK refuses to start without one.
OLLAMA_PLACEHOLDER_KEY = "ollama"

_WINDOW = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-4]):([0-5]\d)$")


class MissingAPIKeyError(RuntimeError):
    """Raised when a model needs an API key that is not set in the environment."""


class ChatMessage(BaseModel):
    """One message in an OpenAI-style chat conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str


class ModelPricing(BaseModel):
    """USD prices per 1M tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_cache_miss_per_1m: NonNegativeFloat
    input_cache_hit_per_1m: NonNegativeFloat
    output_per_1m: NonNegativeFloat


class ThinkingConfig(BaseModel):
    """DeepSeek thinking-mode settings. Thinking is ON by default at DeepSeek, so we always send it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    reasoning_effort: Literal["low", "high", "max"] = "high"


class ModelConfig(BaseModel):
    """Everything needed to call one model and price its responses."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str = Field(min_length=1, description="Logical name used in routing config, e.g. 'deepseek-flash'.")
    provider: Provider
    model_id: str = Field(min_length=1, description="Model identifier sent to the provider API.")
    base_url: str
    api_key_env: str | None = Field(default=None, description="Env var holding the API key; None for Ollama.")
    quality_tier: QualityTier
    typical_latency_ms: PositiveInt
    pricing: ModelPricing
    thinking: ThinkingConfig | None = Field(default=None, description="None = provider has no thinking toggle.")

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {value!r}")
        return value.rstrip("/")

    def has_api_key(self) -> bool:
        """True if the model can be called (Ollama needs no key)."""
        return self.api_key_env is None or bool(os.environ.get(self.api_key_env))

    def resolve_api_key(self) -> str:
        """Return the API key from the environment, or raise :class:`MissingAPIKeyError`."""
        if self.api_key_env is None:
            return OLLAMA_PLACEHOLDER_KEY
        key = os.environ.get(self.api_key_env)
        if not key:
            raise MissingAPIKeyError(
                f"Model {self.name!r} needs {self.api_key_env} — set it in .env (see .env.example)"
            )
        return key


class PriceSchedule(BaseModel):
    """When a provider charges peak vs off-peak prices (all times UTC)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    timezone: Literal["UTC"] = "UTC"
    peak_weekdays: list[Weekday]
    peak_windows: list[str] = Field(description='"HH:MM-HH:MM" ranges; start inclusive, end exclusive.')
    off_peak_multiplier: float = Field(gt=0, le=1)
    note: str = ""

    @field_validator("peak_windows")
    @classmethod
    def _validate_windows(cls, windows: list[str]) -> list[str]:
        for window in windows:
            match = _WINDOW.match(window)
            if not match:
                raise ValueError(f"Bad window {window!r}; expected 'HH:MM-HH:MM'")
            start = int(match.group(1)) * 60 + int(match.group(2))
            end = int(match.group(3)) * 60 + int(match.group(4))
            if end <= start or end > 24 * 60:
                raise ValueError(f"Window {window!r} must end after it starts, within one day")
        return windows


class LLMResponse(BaseModel):
    """Normalised result of one chat completion, identical across providers."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_name: str
    model_id: str
    provider: Provider
    output_text: str
    input_tokens: NonNegativeInt = Field(description="All prompt tokens (cache hits + misses).")
    cached_input_tokens: NonNegativeInt = Field(description="Prompt tokens billed at the cache-hit price.")
    output_tokens: NonNegativeInt = Field(description="Completion tokens, including any reasoning tokens.")
    reasoning_tokens: NonNegativeInt = 0
    latency_ms: NonNegativeFloat = Field(description="Wall time at the provider, including retries.")
    cost_usd: NonNegativeFloat = Field(description="At PEAK prices (conservative).")
    finish_reason: str = Field(description="'stop', 'length', ...; 'unknown' if the provider sent none.")
    attempts: PositiveInt = 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def truncated(self) -> bool:
        """True if the answer was cut off by max_tokens. Such answers are excluded from quality scoring."""
        return self.finish_reason == "length"
