"""Baseline runner and the smoke-test script, end to end with a fake SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autopilot.evaluation.baseline import build_report, run_baseline, summarise
from autopilot.evaluation.smoke_prompts import SMOKE_PROMPTS
from autopilot.llm.client import LLMClient
from autopilot.llm.schemas import ModelConfig
from tests.conftest import FakeOpenAI, bad_request_error, load_script, make_client, make_completion


def test_smoke_prompts_are_well_formed() -> None:
    assert len(SMOKE_PROMPTS) == 10
    assert len({p.prompt_id for p in SMOKE_PROMPTS}) == 10
    assert {p.intuitive_tier for p in SMOKE_PROMPTS} == {1, 2, 3}


async def test_run_baseline_and_summarise(ollama_model: ModelConfig, flash_model: ModelConfig, api_key: str) -> None:
    async def by_model(kwargs: dict[str, Any]) -> Any:
        if kwargs["model"] == "deepseek-flash":
            raise bad_request_error()
        return make_completion("the answer is 1930", finish_reason="length")

    client, _ = make_client(FakeOpenAI([by_model]))
    prompts = list(SMOKE_PROMPTS[:3])
    results = await run_baseline(client, prompts, [ollama_model, flash_model], max_tokens=50)

    assert len(results) == 6
    summary = {s.model_name: s for s in summarise(results, [ollama_model, flash_model])}
    assert summary["local-llama"].n_ok == 3 and summary["local-llama"].n_truncated == 3
    assert summary["deepseek-flash"].n_failed == 3 and summary["deepseek-flash"].mean_latency_ms is None
    # p03's expected answer is "1930"; p01 and p02 expect other strings
    assert (summary["local-llama"].expected_hits, summary["local-llama"].expected_total) == (1, 3)

    report = build_report(results, list(summary.values()), prompts, skipped_models={}, pricing_status="x", max_tokens=50)
    json.dumps(report)  # must be serialisable
    assert report["results"][0]["response"]["truncated"] is True


async def test_smoke_script_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    script = load_script("smoke_test")
    fake = FakeOpenAI([make_completion("Answer: $57")])
    monkeypatch.setattr(
        script, "LLMClient", lambda settings: LLMClient(settings, client_factory=lambda m, s: fake)  # type: ignore[arg-type]
    )
    output = tmp_path / "baseline.json"

    exit_code = await script.main(["--limit", "2", "--output", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text())
    assert len(report["results"]) == 2 * 3  # 2 prompts x 3 models
    assert {s["model_name"] for s in report["summary"]} == {"local-llama", "deepseek-flash", "deepseek-v4-pro"}
    assert report["skipped_models"] == {}
    # warm-up call to Ollama + 6 real calls
    assert len(fake.completions.calls) == 7


async def test_smoke_script_skips_models_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script("smoke_test")
    fake = FakeOpenAI([make_completion()])
    monkeypatch.setattr(
        script, "LLMClient", lambda settings: LLMClient(settings, client_factory=lambda m, s: fake)  # type: ignore[arg-type]
    )
    output = tmp_path / "baseline.json"

    exit_code = await script.main(["--limit", "1", "--no-warmup", "--output", str(output)])

    report = json.loads(output.read_text())
    assert exit_code == 0
    assert set(report["skipped_models"]) == {"deepseek-flash", "deepseek-v4-pro"}
    assert len(report["results"]) == 1
