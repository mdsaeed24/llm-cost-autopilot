"""Model catalogue: config/models.yaml parses, validates and honours env overrides."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autopilot.llm.catalog import load_catalog
from autopilot.llm.schemas import MissingAPIKeyError, ModelConfig, ModelPricing


def test_repo_catalog_has_three_tiers_in_order() -> None:
    catalog = load_catalog()
    assert [m.name for m in catalog.by_quality()] == ["local-llama", "deepseek-flash", "deepseek-v4-pro"]
    assert [m.quality_tier for m in catalog.by_quality()] == ["low", "medium", "high"]


def test_repo_catalog_defaults() -> None:
    catalog = load_catalog()
    llama, flash, pro = (catalog.get(n) for n in ("local-llama", "deepseek-flash", "deepseek-v4-pro"))
    assert llama.model_id == "llama3.2:3b"
    assert llama.thinking is None
    assert llama.pricing.output_per_1m == 0
    assert flash.thinking is not None and flash.thinking.enabled is False
    assert pro.thinking is not None and pro.thinking.enabled is False
    assert flash.pricing.input_cache_miss_per_1m == pytest.approx(0.30)
    assert pro.pricing.output_per_1m == pytest.approx(3.96)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:1b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1/")
    monkeypatch.setenv("DEEPSEEK_PRO_THINKING", "true")
    catalog = load_catalog()
    assert catalog.get("local-llama").model_id == "llama3.2:1b"
    assert catalog.get("local-llama").base_url == "http://host.docker.internal:11434/v1"
    assert catalog.get("deepseek-v4-pro").thinking.enabled is True  # type: ignore[union-attr]


def test_unknown_model_lists_known_names() -> None:
    with pytest.raises(KeyError, match="deepseek-flash"):
        load_catalog().get("gpt-9")


def test_negative_price_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelPricing(input_cache_miss_per_1m=-1, input_cache_hit_per_1m=0, output_per_1m=0)


def test_bad_base_url_rejected(ollama_model: ModelConfig) -> None:
    with pytest.raises(ValidationError, match="base_url"):
        ModelConfig.model_validate({**ollama_model.model_dump(), "base_url": "localhost:11434"})


def test_api_key_resolution(ollama_model: ModelConfig, flash_model: ModelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ollama_model.has_api_key()
    assert ollama_model.resolve_api_key() == "ollama"
    assert not flash_model.has_api_key()
    with pytest.raises(MissingAPIKeyError, match="DEEPSEEK_API_KEY"):
        flash_model.resolve_api_key()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert flash_model.resolve_api_key() == "sk-test"
