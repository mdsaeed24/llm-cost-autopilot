"""Send the same prompts to every model and summarise cost, latency and truncation.

Used by ``scripts/smoke_test.py`` to validate the unified interface and build intuition.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from autopilot.evaluation.smoke_prompts import SmokePrompt
from autopilot.llm.client import LLMClient, LLMRequestError
from autopilot.llm.schemas import ChatMessage, LLMResponse, ModelConfig


class BaselineResult(BaseModel):
    """One (prompt, model) outcome: either a response or an error."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    prompt_id: str
    kind: str
    model_name: str
    response: LLMResponse | None = None
    error: str | None = None
    contains_expected: bool | None = None


class ModelSummary(BaseModel):
    """Aggregates for one model across all prompts."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    model_id: str
    quality_tier: str
    n_ok: int
    n_failed: int
    n_truncated: int
    total_cost_usd: float
    mean_latency_ms: float | None
    median_latency_ms: float | None
    max_latency_ms: float | None
    mean_output_tokens: float | None
    projected_cost_per_1k_usd: float | None
    expected_hits: int
    expected_total: int


async def _run_one(client: LLMClient, prompt: SmokePrompt, model: ModelConfig, max_tokens: int) -> BaselineResult:
    try:
        response = await client.send(
            [ChatMessage(role="user", content=prompt.prompt)], model, max_tokens=max_tokens
        )
    except LLMRequestError as error:
        return BaselineResult(prompt_id=prompt.prompt_id, kind=prompt.kind, model_name=model.name, error=str(error))
    contains = None
    if prompt.expected is not None:
        contains = prompt.expected.lower() in response.output_text.lower()
    return BaselineResult(
        prompt_id=prompt.prompt_id,
        kind=prompt.kind,
        model_name=model.name,
        response=response,
        contains_expected=contains,
    )


async def run_baseline(
    client: LLMClient,
    prompts: Sequence[SmokePrompt],
    models: Sequence[ModelConfig],
    *,
    max_tokens: int,
) -> list[BaselineResult]:
    """Run every prompt against every model concurrently (bounded by the client's semaphores)."""
    tasks = [_run_one(client, prompt, model, max_tokens) for model in models for prompt in prompts]
    return list(await asyncio.gather(*tasks))


def summarise(results: Sequence[BaselineResult], models: Sequence[ModelConfig]) -> list[ModelSummary]:
    """Per-model totals and latency statistics."""
    summaries: list[ModelSummary] = []
    for model in models:
        rows = [r for r in results if r.model_name == model.name]
        ok = [r.response for r in rows if r.response is not None]
        latencies = [r.latency_ms for r in ok]
        total_cost = sum(r.cost_usd for r in ok)
        checked = [r for r in rows if r.contains_expected is not None]
        summaries.append(
            ModelSummary(
                model_name=model.name,
                model_id=model.model_id,
                quality_tier=model.quality_tier,
                n_ok=len(ok),
                n_failed=len(rows) - len(ok),
                n_truncated=sum(r.truncated for r in ok),
                total_cost_usd=total_cost,
                mean_latency_ms=statistics.fmean(latencies) if latencies else None,
                median_latency_ms=statistics.median(latencies) if latencies else None,
                max_latency_ms=max(latencies) if latencies else None,
                mean_output_tokens=statistics.fmean(r.output_tokens for r in ok) if ok else None,
                projected_cost_per_1k_usd=(total_cost / len(ok) * 1000) if ok else None,
                expected_hits=sum(bool(r.contains_expected) for r in checked),
                expected_total=len(checked),
            )
        )
    return summaries


def build_report(
    results: Sequence[BaselineResult],
    summaries: Sequence[ModelSummary],
    prompts: Sequence[SmokePrompt],
    *,
    skipped_models: dict[str, str],
    pricing_status: str,
    max_tokens: int,
) -> dict[str, Any]:
    """JSON-serialisable report written to ``data/baseline_comparison.json``."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "pricing_basis": "DeepSeek PEAK prices (conservative; off-peak is 50% cheaper)",
        "pricing_status_at_run": pricing_status,
        "max_tokens": max_tokens,
        "skipped_models": skipped_models,
        "prompts": [p.model_dump() for p in prompts],
        "summary": [s.model_dump() for s in summaries],
        "results": [r.model_dump(mode="json") for r in results],
    }
