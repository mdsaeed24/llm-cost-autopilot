"""Load test against the running API, in evaluation mode (every routed answer is verified).

Flow: estimate cost (using the real classifier locally, free) → send prompts with a
concurrency cap → wait for the background worker to verify everything → fetch the honest
stats for exactly this run (``/v1/stats?mode=evaluation&since=<start>``) → JSON + Markdown.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from autopilot.dataset.schemas import LoadtestPrompt
from autopilot.llm.catalog import ModelCatalog
from autopilot.llm.schemas import ModelPricing
from autopilot.routing.router import Router

# Token assumptions for the up-front estimate (from the Phase 1 baseline: 200-320 output tokens
# on average; we assume more, to stay conservative).
EST_OUTPUT_TOKENS = 400
EST_JUDGE_OVERHEAD_TOKENS = 300   # judge system prompt + criterion
EST_JUDGE_OUTPUT_TOKENS = 120


class LoadtestAbort(RuntimeError):
    """The environment is not ready for a load test (e.g. API not in evaluation mode)."""


# ---------- estimate ----------

class CostEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: int
    routed_tiers: dict[int, int]
    serve_usd: float
    verification_usd: float
    total_usd: float
    offpeak_total_usd: float


def _cost(pricing: ModelPricing, input_tokens: float, output_tokens: float) -> float:
    return (input_tokens * pricing.input_cache_miss_per_1m + output_tokens * pricing.output_per_1m) / 1_000_000


def estimate_cost(
    prompts: Sequence[LoadtestPrompt], router: Router, catalog: ModelCatalog, *, offpeak_multiplier: float
) -> CostEstimate:
    """Classify every prompt locally (free) and price the predicted traffic, including verification."""
    tiers: Counter[int] = Counter()
    serve = verification = 0.0
    top = catalog.get(router.config.tiers[3]).pricing
    for prompt in prompts:
        tier = router.decide(prompt.prompt).routed_tier
        tiers[tier] += 1
        input_tokens = len(prompt.prompt) / 4 + 10
        serve += _cost(catalog.get(router.config.tiers[tier]).pricing, input_tokens, EST_OUTPUT_TOKENS)
        if tier < 3:  # evaluation mode: Tier-3 reference + judge for every downgraded request
            judge_input = input_tokens + 2 * EST_OUTPUT_TOKENS + EST_JUDGE_OVERHEAD_TOKENS
            verification += _cost(top, input_tokens, EST_OUTPUT_TOKENS) + _cost(top, judge_input, EST_JUDGE_OUTPUT_TOKENS)
    total = serve + verification
    return CostEstimate(
        prompts=len(prompts), routed_tiers=dict(sorted(tiers.items())), serve_usd=serve,
        verification_usd=verification, total_usd=total, offpeak_total_usd=total * offpeak_multiplier,
    )


# ---------- running ----------

class ClientResult(BaseModel):
    """One request as seen by the load-test client."""

    model_config = ConfigDict(extra="forbid")

    loadtest_id: str
    ok: bool
    status_code: int | None
    error: str | None = None
    chosen_model: str | None = None
    chosen_tier: int | None = None
    routed_up: bool | None = None
    escalated: bool | None = None
    cost_usd: float | None = None
    client_latency_ms: float
    finish_reason: str | None = None


async def check_api(client: httpx.AsyncClient) -> dict[str, Any]:
    """The API must be up and in evaluation mode (otherwise only ~15% would be verified)."""
    try:
        response = await client.get("/health")
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise LoadtestAbort(f"API not reachable at {client.base_url} ({error}). Start it with `make up-eval`.") from error
    health = response.json()
    if health.get("mode") != "evaluation":
        raise LoadtestAbort(
            f"API is in {health.get('mode')!r} mode; the load test needs evaluation mode (verify every request). "
            "Restart with `make up-eval`."
        )
    return health  # type: ignore[no-any-return]


async def send_all(
    client: httpx.AsyncClient,
    prompts: Sequence[LoadtestPrompt],
    *,
    concurrency: int,
    on_done: Callable[[ClientResult], None] | None = None,
) -> list[ClientResult]:
    """POST every prompt to /v1/completions with at most ``concurrency`` in flight."""
    gate = asyncio.Semaphore(concurrency)

    async def one(prompt: LoadtestPrompt) -> ClientResult:
        async with gate:
            started = time.perf_counter()
            try:
                response = await client.post(
                    "/v1/completions", json={"messages": [{"role": "user", "content": prompt.prompt}]}
                )
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    error = response.json().get("error", {}).get("message", response.text[:200])
                    result = ClientResult(loadtest_id=prompt.loadtest_id, ok=False, status_code=response.status_code,
                                          error=error, client_latency_ms=elapsed)
                else:
                    body = response.json()
                    meta = body["autopilot"]
                    result = ClientResult(
                        loadtest_id=prompt.loadtest_id, ok=True, status_code=200,
                        chosen_model=meta["chosen_model"], chosen_tier=meta["chosen_tier"],
                        routed_up=meta["routed_up"], escalated=meta["escalated"], cost_usd=meta["cost_usd"],
                        client_latency_ms=elapsed, finish_reason=body["choices"][0]["finish_reason"],
                    )
            except httpx.HTTPError as error:
                result = ClientResult(loadtest_id=prompt.loadtest_id, ok=False, status_code=None,
                                      error=f"{type(error).__name__}: {error}",
                                      client_latency_ms=(time.perf_counter() - started) * 1000)
        if on_done:
            on_done(result)
        return result

    return list(await asyncio.gather(*(one(p) for p in prompts)))


async def wait_for_verification(
    client: httpx.AsyncClient,
    *,
    timeout_s: float,
    poll_s: float = 10.0,
    on_poll: Callable[[dict[str, int]], None] | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> tuple[bool, dict[str, int]]:
    """Poll /health until no verification job is queued or running. Returns (drained, queue counts)."""
    deadline = time.monotonic() + timeout_s
    while True:
        queue: dict[str, int] = (await client.get("/health")).json().get("queue", {})
        if on_poll:
            on_poll(queue)
        if queue.get("queued", 0) + queue.get("running", 0) == 0:
            return True, queue
        if time.monotonic() >= deadline:
            return False, queue
        await sleep(poll_s)


async def fetch_stats(client: httpx.AsyncClient, since: datetime) -> dict[str, Any]:
    response = await client.get("/v1/stats", params={"mode": "evaluation", "since": since.isoformat()})
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


# ---------- report ----------

def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1])


def build_report(
    *,
    results: Sequence[ClientResult],
    stats: dict[str, Any],
    estimate: CostEstimate,
    meta: dict[str, Any],
    offpeak_multiplier: float,
    production_sample_rate: float,
) -> dict[str, Any]:
    """Everything the JSON report contains. All headline numbers come from /v1/stats (one definition).

    Evaluation mode verifies 100% of downgraded answers, so its net savings are the worst case.
    We also report a clearly labelled *projection* at the production sample rate: verification
    cost scales linearly with the share of requests verified; serving cost and parity do not change.
    """
    summary = stats["summary"]
    baseline = summary["baseline_cost_usd"]
    projected_total = summary["serve_cost_usd"] + production_sample_rate * summary["verification_cost_usd"]
    projected_net = (baseline - projected_total) / baseline if baseline else None
    ok = [r for r in results if r.ok]
    client_latency: dict[str, dict[str, float | None]] = {}
    for tier in sorted({r.chosen_tier for r in ok if r.chosen_tier is not None}):
        values = [r.client_latency_ms for r in ok if r.chosen_tier == tier]
        client_latency[f"tier_{tier}"] = {"n": len(values), "p50_ms": _percentile(values, 50), "p95_ms": _percentile(values, 95)}
    return {
        **meta,
        "headline": stats["headline"],
        "run": {
            **meta.get("run", {}),
            "sent": len(results),
            "ok": len(ok),
            "failed": len(results) - len(ok),
            "failures": Counter(r.error or "unknown" for r in results if not r.ok).most_common(5),
        },
        "costs": {
            "serve_usd": summary["serve_cost_usd"],
            "verification_usd": summary["verification_cost_usd"],
            "total_usd": summary["total_cost_usd"],
            "all_tier3_baseline_usd": summary["baseline_cost_usd"],
            "baseline_measured_share": summary["baseline_measured_share"],
            "gross_savings_pct": summary["gross_savings_pct"],
            "net_savings_pct": summary["net_savings_pct"],
            "verification_overhead_pct_of_baseline": summary["verification_overhead_pct"],
            "offpeak_equivalent_total_usd": summary["total_cost_usd"] * offpeak_multiplier,
            "projection": {
                "sample_rate": production_sample_rate,
                "total_usd": projected_total,
                "net_savings_pct": projected_net,
                "note": "verification cost scaled linearly to the production sample rate; serving cost unchanged",
            },
            "estimate": estimate.model_dump(),
        },
        "quality": summary["quality"],
        "routing": {
            "distribution": summary["routing_share"],
            "routed_up_rate": summary["routed_up_rate"],
            "failover_rate": summary["failover_rate"],
            "escalation_rate": summary["escalation_rate"],
            "escalations_async": summary["escalations_async"],
            "escalations_sync": summary["escalations_sync"],
        },
        "per_tier": summary["tiers"],
        "client_latency": client_latency,
        "stats": stats,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.1f}s"


def render_markdown(report: dict[str, Any]) -> str:
    """Human summary. Savings always appear with quality parity and verification overhead."""
    c, q, r = report["costs"], report["quality"], report["routing"]
    lines = [
        "# Load test report",
        "",
        f"**{report['headline']}**",
        "",
        f"- Run: {report['run']['sent']} prompts sent, {report['run']['ok']} answered, {report['run']['failed']} failed · "
        f"{report['run'].get('duration_s', 0):.0f}s · concurrency {report['run'].get('concurrency')} · "
        f"classifier {report.get('classifier_version')} · mode {report.get('mode')}",
        f"- Pricing: DeepSeek **peak** prices throughout (conservative). {report.get('pricing_status_at_start', '')}",
        "",
        "## Cost vs. everything on Tier 3",
        "",
        "| | USD |",
        "|---|---|",
        f"| All-Tier-3 baseline | ${c['all_tier3_baseline_usd']:.4f} |",
        f"| Autopilot — answering | ${c['serve_usd']:.4f} |",
        f"| Autopilot — verification (Tier-3 reference + judge) | ${c['verification_usd']:.4f} |",
        f"| **Autopilot — total** | **${c['total_usd']:.4f}** |",
        f"| Same run at off-peak prices | ${c['offpeak_equivalent_total_usd']:.4f} |",
        "",
        f"- **Net savings (after verification): {_pct(c['net_savings_pct'])}** · gross (before verification): "
        f"{_pct(c['gross_savings_pct'])} · verification overhead: {_pct(c['verification_overhead_pct_of_baseline'])} of baseline",
        f"- **Projected net savings in production ({_pct(c['projection']['sample_rate'])} of requests verified): "
        f"{_pct(c['projection']['net_savings_pct'])}** — projection: verification cost scaled linearly, same routing.",
        f"- **Quality parity: {_pct(q['parity'])}** (95% CI {_pct(q['ci_low'])}–{_pct(q['ci_high'])}, "
        f"n={q['judged']} randomly verified downgraded answers)",
        f"- Baseline measured (real Tier-3 cost) for {_pct(c['baseline_measured_share'])} "
        "of requests; the rest estimated from token counts.",
        f"- Up-front estimate was ${c['estimate']['total_usd']:.2f} (actual ${c['total_usd']:.2f}).",
        "- Evaluation mode verifies **every** downgraded request, so the verification overhead here is the "
        "worst case. Production samples ~15%.",
        "",
        "## Quality",
        "",
        f"- Judged: {q['judged']} · passed: {q['passed']} · parity {_pct(q['parity'])}",
        f"- Excluded from parity (counted, not hidden): truncated routed answers {q['excluded_routed_truncated']}, "
        f"truncated references {q['excluded_reference_truncated']}, unparseable judge replies "
        f"{q['excluded_judge_unparseable']}, failed verification jobs {q['verification_failed_jobs']}, "
        f"still pending {q['pending']}",
        "",
        "## Routing",
        "",
        "| Model | Share of requests |",
        "|---|---|",
        *[f"| {model} | {_pct(share)} |" for model, share in sorted(r["distribution"].items(), key=lambda kv: -kv[1])],
        "",
        f"- Routed up for low confidence: {_pct(r['routed_up_rate'])} · failover (model down): {_pct(r['failover_rate'])}",
        f"- Escalation rate: {_pct(r['escalation_rate'])} ({r['escalations_async']} async logged failures, "
        f"{r['escalations_sync']} sync replacements)",
        "",
        "## Per tier",
        "",
        "| Tier | Requests | Truncated (excluded from parity) | P50 latency | P95 latency |",
        "|---|---|---|---|---|",
        *[f"| {t['tier']} | {t['requests']} | {t['truncated']} ({_pct(t['truncation_rate'])}) | "
          f"{_ms(t['p50_latency_ms'])} | {_ms(t['p95_latency_ms'])} |" for t in report["per_tier"]],
        "",
        "Latency is server-side end-to-end (classification + model call). Tier 1 runs on a laptop via Ollama.",
        "",
        "## Caveats",
        "",
        "- The judge is the same model family as Tier 3 and may prefer its own style (self-preference bias); "
        "answers are blinded and randomly ordered to reduce it.",
        "- Prompts come from Databricks Dolly 15k (CC BY-SA 3.0): a uniform random pool, disjoint from training data.",
        "",
    ]
    return "\n".join(lines)


def now_utc() -> datetime:
    return datetime.now(UTC)
