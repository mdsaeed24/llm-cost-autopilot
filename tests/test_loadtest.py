"""Load test: estimate, safety checks, confirmation, peak warning, end-to-end report (fake API)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from autopilot.analytics.metrics import load_frames, summarise
from autopilot.dataset.io import write_jsonl
from autopilot.dataset.schemas import LoadtestPrompt
from autopilot.evaluation.loadtest import LoadtestAbort, check_api, estimate_cost, wait_for_verification
from autopilot.llm.catalog import load_catalog
from autopilot.llm.pricing import PeakStatus
from autopilot.routing.config import ConfidenceRule, RoutingConfig
from autopilot.routing.router import Router
from autopilot.storage.db import Database
from tests.conftest import load_script
from tests.test_metrics_honest import db as metrics_db  # noqa: F401 - fixture reuse
from tests.test_verification_flow import Stub

PROMPTS = [LoadtestPrompt(loadtest_id=f"lt{i:04d}", prompt=f"Question number {i}?") for i in range(1, 4)]
ROUTING = RoutingConfig(tiers={1: "local-llama", 2: "deepseek-flash", 3: "deepseek-v4-pro"},
                        confidence=ConfidenceRule(enabled=False, threshold=0.5))


def test_estimate_prices_verification_only_for_downgraded_requests() -> None:
    catalog = load_catalog()
    tier1 = estimate_cost(PROMPTS, Router(Stub(1), ROUTING, catalog), catalog, offpeak_multiplier=0.5)
    tier3 = estimate_cost(PROMPTS, Router(Stub(3), ROUTING, catalog), catalog, offpeak_multiplier=0.5)
    assert tier1.routed_tiers == {1: 3} and tier1.serve_usd == 0 and tier1.verification_usd > 0
    assert tier3.routed_tiers == {3: 3} and tier3.verification_usd == 0 and tier3.serve_usd > 0
    assert tier1.offpeak_total_usd == pytest.approx(tier1.total_usd / 2)


class FakeApi:
    """Stands in for the running API."""

    def __init__(self, stats: dict[str, Any], *, mode: str = "evaluation") -> None:
        self.mode = mode
        self.stats = stats
        self.health_calls = 0
        self.posts: list[dict[str, Any]] = []
        self.stats_params: dict[str, str] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_calls += 1
            queue = {"queued": 2, "done": 1} if self.health_calls == 2 else {"done": 3}
            return httpx.Response(200, json={"status": "ok", "mode": self.mode, "classifier_version": "v2",
                                             "escalation_mode": "async", "queue": queue})
        if request.url.path == "/v1/completions":
            body = json.loads(request.content)
            self.posts.append(body)
            if "3" in body["messages"][0]["content"]:
                return httpx.Response(502, json={"error": {"type": "upstream_error", "message": "all tiers down"}})
            tier = len(self.posts) % 2 + 1
            return httpx.Response(200, json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "autopilot": {"chosen_model": "local-llama" if tier == 1 else "deepseek-flash", "chosen_tier": tier,
                              "routed_up": False, "escalated": False, "cost_usd": 0.0001},
            })
        if request.url.path == "/v1/stats":
            self.stats_params = dict(request.url.params)
            return httpx.Response(200, json=self.stats)
        return httpx.Response(404)


@pytest.fixture
def stats(metrics_db: Database) -> dict[str, Any]:  # noqa: F811
    pro = load_catalog().get("deepseek-v4-pro")
    summary = summarise(load_frames(metrics_db), tier3_model=pro.name, tier3_pricing=pro.pricing)
    return {"headline": summary.headline(), "mode": "evaluation", "since_hours": None,
            "summary": summary.model_dump(mode="json")}


async def test_check_api_requires_evaluation_mode(stats: dict[str, Any]) -> None:
    async with httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(FakeApi(stats, mode="normal"))) as c:
        with pytest.raises(LoadtestAbort, match="make up-eval"):
            await check_api(c)

    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(down)) as c:
        with pytest.raises(LoadtestAbort, match="not reachable"):
            await check_api(c)


async def test_wait_for_verification_drains_or_times_out(stats: dict[str, Any]) -> None:
    api = FakeApi(stats)
    api.health_calls = 1  # next poll reports work still queued
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async with httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(api)) as c:
        drained, queue = await wait_for_verification(c, timeout_s=60, poll_s=5, sleep=fake_sleep)
        assert drained and queue == {"done": 3} and slept == [5]

        api.health_calls = 1
        drained, queue = await wait_for_verification(c, timeout_s=0, poll_s=5, sleep=fake_sleep)
        assert not drained and queue["queued"] == 2


def _args(tmp_path: Path, *extra: str) -> list[str]:
    write_jsonl(tmp_path / "pool.jsonl", PROMPTS)
    return ["--prompts", str(tmp_path / "pool.jsonl"), "--out", str(tmp_path / "report.json"),
            "--md", str(tmp_path / "report.md"), "--api-url", "http://api", "--limit", "3", *extra]


async def test_full_run_writes_honest_report(tmp_path: Path, stats: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script("loadtest")
    monkeypatch.setattr("autopilot.evaluation.loadtest.asyncio.sleep", _no_sleep)
    api = FakeApi(stats)
    code = await script.main(_args(tmp_path, "--yes"), transport=httpx.MockTransport(api))

    assert code == 3  # one request failed (502) → non-zero exit, but the report is still written
    assert len(api.posts) == 3 and all(p == {"messages": p["messages"]} for p in api.posts)  # no model chosen
    assert api.stats_params["mode"] == "evaluation" and "since" in api.stats_params

    report = json.loads((tmp_path / "report.json").read_text())
    assert (report["run"]["sent"], report["run"]["ok"], report["run"]["failed"]) == (3, 2, 1)
    assert report["costs"]["net_savings_pct"] is not None and report["quality"]["judged"] == 2
    c = report["costs"]
    projected = c["serve_usd"] + 0.15 * c["verification_usd"]
    assert c["projection"]["total_usd"] == pytest.approx(projected)
    assert c["projection"]["net_savings_pct"] == pytest.approx(1 - projected / c["all_tier3_baseline_usd"])
    assert c["net_savings_pct"] < c["projection"]["net_savings_pct"] <= c["gross_savings_pct"]
    assert report["per_tier"][0]["truncation_rate"] is not None
    assert set(report["client_latency"]) == {"tier_1", "tier_2"}

    md = (tmp_path / "report.md").read_text()
    for fragment in ("Net savings (after verification)", "Projected net savings in production", "Quality parity", "verification overhead",
                     "Truncated (excluded from parity)", "P95 latency", "off-peak", "Escalation rate"):
        assert fragment in md


async def _no_sleep(_: float) -> None:
    return None


async def test_declining_the_estimate_sends_nothing(tmp_path: Path, stats: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script("loadtest")
    monkeypatch.setattr("builtins.input", lambda _q: "n")
    api = FakeApi(stats)
    assert await script.main(_args(tmp_path), transport=httpx.MockTransport(api)) == 1
    assert api.posts == [] and api.health_calls == 0


async def test_wrong_mode_aborts_before_sending(tmp_path: Path, stats: dict[str, Any]) -> None:
    script = load_script("loadtest")
    api = FakeApi(stats, mode="normal")
    assert await script.main(_args(tmp_path, "--yes"), transport=httpx.MockTransport(api)) == 2
    assert api.posts == []


async def test_peak_hours_warning(tmp_path: Path, stats: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    script = load_script("loadtest")
    peak = PeakStatus(at=datetime(2026, 1, 5, 2, tzinfo=UTC), is_peak=True, price_multiplier=1.0,
                      next_change=datetime(2026, 1, 5, 4, tzinfo=UTC))
    monkeypatch.setattr(script, "peak_status", lambda _schedule: peak)
    monkeypatch.setattr("builtins.input", lambda _q: "n")
    await script.main(_args(tmp_path), transport=httpx.MockTransport(FakeApi(stats)))
    out = capsys.readouterr().out
    assert "PEAK hours" in out and "2×" in out and "Mon 04:00 UTC" in out
