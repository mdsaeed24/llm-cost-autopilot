"""Config loading: env expansion and settings validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autopilot.settings import ConfigError, expand_env, load_settings, load_yaml


def test_expand_env_uses_default_when_unset() -> None:
    assert expand_env("${NOT_SET_VAR:-llama3.2:3b}") == "llama3.2:3b"


def test_expand_env_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:1b")
    assert expand_env("${OLLAMA_MODEL:-llama3.2:3b}") == "llama3.2:1b"


def test_expand_env_treats_empty_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "")
    assert expand_env("${OLLAMA_MODEL:-fallback}") == "fallback"


def test_expand_env_without_default_raises() -> None:
    with pytest.raises(ConfigError, match="NOT_SET_VAR"):
        expand_env("${NOT_SET_VAR}")


def test_expand_env_recurses_and_keeps_non_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "1")
    assert expand_env({"a": ["${X}", 2], "b": {"c": None}}) == {"a": ["1", 2], "b": {"c": None}}


def test_repo_settings_load() -> None:
    settings = load_settings()
    assert settings.client.concurrency_for("ollama") == 2
    assert settings.client.concurrency_for("deepseek") == 5
    assert settings.client.max_output_tokens == 2048


def test_unknown_settings_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("client:\n  concurency_default: 3\n")  # typo
    with pytest.raises(ValidationError):
        load_settings(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")
