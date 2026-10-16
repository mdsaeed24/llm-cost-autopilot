"""Load YAML configuration (with ``${VAR:-default}`` env expansion) into typed settings."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or references an unset variable."""


def config_dir() -> Path:
    """Directory holding the YAML config files (override with ``AUTOPILOT_CONFIG_DIR``)."""
    return Path(os.environ.get("AUTOPILOT_CONFIG_DIR", PROJECT_ROOT / "config"))


def load_env() -> None:
    """Load ``.env`` from the project root. Real environment variables always win."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def expand_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` in strings.

    Like the shell's ``:-``, an empty variable counts as unset and the default is used.
    """
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def _substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        env_value = os.environ.get(name)
        if env_value:
            return env_value
        if default is not None:
            return default
        raise ConfigError(f"Environment variable {name!r} is not set and has no default")

    return _ENV_PLACEHOLDER.sub(_substitute, value)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` and expand environment placeholders."""
    load_env()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at the top of {path}")
    return expand_env(data)


class ClientSettings(BaseModel):
    """How the LLM client talks to providers: concurrency, retries, timeouts, token cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency_default: PositiveInt = 5
    concurrency_per_provider: dict[str, PositiveInt] = Field(default_factory=dict)
    max_attempts: PositiveInt = 4
    backoff_base_s: PositiveFloat = 1.0
    backoff_max_s: PositiveFloat = 30.0
    request_timeout_s: PositiveFloat = 120.0
    max_output_tokens: PositiveInt = 2048

    def concurrency_for(self, provider: str) -> int:
        """Max in-flight requests allowed for ``provider``."""
        return self.concurrency_per_provider.get(provider, self.concurrency_default)


class EmbeddingSettings(BaseModel):
    """Where to reach the embedding model. The model *name* lives in the trained classifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "http://localhost:11434/v1"
    timeout_s: PositiveFloat = 30.0
    batch_size: PositiveInt = 32
    cache_path: str | None = "data/raw/embedding_cache.sqlite"


class DatabaseSettings(BaseModel):
    """SQLite file shared by the API, the worker and the dashboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = "data/autopilot.db"

    def resolved_path(self) -> Path:
        """Relative paths are relative to the project root."""
        path = Path(self.path)
        return path if path.is_absolute() else PROJECT_ROOT / path


RunMode = Literal["normal", "evaluation"]
EscalationMode = Literal["async", "sync"]


class VerificationSettings(BaseModel):
    """Background quality checks: how many requests to verify, and what to do about failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RunMode = "normal"
    sample_rate_normal: float = Field(default=0.15, ge=0, le=1)
    sample_rate_evaluation: float = Field(default=1.0, ge=0, le=1)
    sample_rate_override: float | None = Field(default=None, ge=0, le=1)
    escalation_mode: EscalationMode = "async"
    sync_confidence_below: float = Field(default=0.5, ge=0, le=1)
    judge_max_tokens: PositiveInt = 400
    worker_concurrency: PositiveInt = 3
    worker_poll_interval_s: PositiveFloat = 1.0
    job_max_attempts: PositiveInt = 3
    job_visibility_timeout_s: PositiveFloat = 600.0

    @field_validator("sample_rate_override", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if value in ("", None) else value

    @property
    def sample_rate(self) -> float:
        """Fraction of eligible requests to verify (override > mode default)."""
        if self.sample_rate_override is not None:
            return self.sample_rate_override
        return self.sample_rate_evaluation if self.mode == "evaluation" else self.sample_rate_normal


class ApiSettings(BaseModel):
    """HTTP API limits and (optional) authentication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_messages: PositiveInt = 50
    max_total_chars: PositiveInt = 24_000
    max_tokens_cap: PositiveInt = 4096
    api_key: str | None = Field(default=None, description="If set, every request needs 'Authorization: Bearer <key>'.")

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if value in ("", None) else value


class AppSettings(BaseModel):
    """Top-level application settings (``config/settings.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client: ClientSettings = Field(default_factory=ClientSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


def load_settings(path: Path | None = None) -> AppSettings:
    """Load ``config/settings.yaml`` (or ``path``) into :class:`AppSettings`."""
    return AppSettings.model_validate(load_yaml(path or config_dir() / "settings.yaml"))
