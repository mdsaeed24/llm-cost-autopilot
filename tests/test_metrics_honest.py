"""Honest metrics: baseline, savings, parity (random sample only), exclusions, per-tier numbers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from autopilot.analytics.metrics import (
    daily_costs,
    daily_escalations,
    load_frames,
    quality_distribution,
    routing_distribution,
    summarise,
    wilson_interval,
)
from autopilot.llm.catalog import load_catalog
from autopilot.storage.db import Database, utc_now
from autopilot.storage.records import EscalationRecord, VerificationRecord
from autopilot.storage.repository import Repository
from tests.test_storage import request

PRO = load_catalog().get("deepseek-v4-pro")
EST = lambda i, o: (i * 1.32 + o * 3.96) / 1e6  # noqa: E731 - Tier-3 price for an estimated baseline


def verification(rid: str, *, ref: float, judge: float = 0.0, passed: bool | None, agreement: int | None = None,
                 source: str = "sampled", excluded: str | None = None) -> VerificationRecord:
    return VerificationRecord(
        request_id=rid, created_at=utc_now(), source=source, reference_model="deepseek-v4-pro", reference_output="r",
        reference_cost_usd=ref, reference_latency_ms=1000, reference_truncated=excluded == "reference_truncated",
        judge_model="deepseek-v4-pro" if judge else None, judge_cost_usd=judge, agreement=agreement,
        rule="qa_classification", pass_threshold=4, passed=passed, excluded_reason=excluded,
    )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "m.db")
    repo = Repository(database)
    llama = dict(served_model="local-llama", served_tier=1, routed_tier=1, input_tokens=1000, output_tokens=500,
                 serve_cost_usd=0.0, latency_ms=1000)
    repo.insert_request(request("r1", **llama, verification_status="queued"))
    repo.save_verification(verification("r1", ref=0.004, judge=0.001, passed=True, agreement=5))
    repo.insert_request(request("r2", **{**llama, "latency_ms": 3000}, verification_status="queued"))
    repo.save_verification(
        verification("r2", ref=0.003, judge=0.001, passed=False, agreement=2),
        escalation=EscalationRecord(request_id="r2", created_at=utc_now(), mode="async", original_model="local-llama",
                                    escalated_model="deepseek-v4-pro", original_cost_usd=0, escalated_cost_usd=0.003,
                                    cost_delta_usd=0.003, judge_score=2, quality_gap=2, answer_replaced=False),
    )
    repo.insert_request(request("r3", served_model="deepseek-flash", served_tier=2, routed_tier=2, routed_up=True,
                                input_tokens=1000, output_tokens=1000, serve_cost_usd=0.0015, latency_ms=2000,
                                verification_status="not_sampled"))
    repo.insert_request(request("r4", served_model="deepseek-v4-pro", served_tier=3, routed_tier=3, input_tokens=500,
                                output_tokens=1000, serve_cost_usd=0.006, latency_ms=5000,
                                verification_status="skipped_top_tier"))
    repo.insert_request(request("r5", **{**llama, "input_tokens": 100, "output_tokens": 2048}, truncated=True,
                                finish_reason="length", verification_status="excluded_truncated"))
    repo.insert_request(request("r6", status="error", error="boom", served_model=None, served_tier=None,
                                verification_status="error"))
    repo.insert_request(request("r7", **llama, verification_status="queued"))
    repo.save_verification(verification("r7", ref=0.002, judge=0.001, passed=True, agreement=5, source="sync"))
    repo.insert_request(request("r8", **llama, verification_status="queued"))
    repo.save_verification(verification("r8", ref=0.002, passed=None, excluded="reference_truncated"))
    return database


def test_summary_numbers(db: Database) -> None:
    s = summarise(load_frames(db), tier3_model=PRO.name, tier3_pricing=PRO.pricing)
    assert (s.requests, s.errors) == (8, 1)
    assert s.serve_cost_usd == pytest.approx(0.0075)
    assert s.verification_cost_usd == pytest.approx(0.005 + 0.004 + 0.003 + 0.002)
    baseline = 0.004 + 0.003 + EST(1000, 1000) + 0.006 + EST(100, 2048) + 0.002 + 0.002
    assert s.baseline_cost_usd == pytest.approx(baseline)
    assert s.baseline_measured_share == pytest.approx(5 / 7)  # r3 and r5 are estimates
    assert s.gross_savings_usd == pytest.approx(baseline - 0.0075)
    assert s.net_savings_usd == pytest.approx(baseline - 0.0075 - 0.014)
    assert s.net_savings_pct == pytest.approx((baseline - 0.0215) / baseline)

    q = s.quality
    assert (q.judged, q.passed, q.parity) == (2, 1, 0.5)  # r7 (sync) is NOT in parity
    assert (q.sync_judged, q.sync_passed) == (1, 1)
    assert (q.excluded_routed_truncated, q.excluded_reference_truncated, q.excluded_judge_unparseable) == (1, 1, 0)
    assert not q.reliable and q.ci_low is not None and q.ci_low < 0.5 < q.ci_high  # type: ignore[operator]

    assert (s.escalations_async, s.escalations_sync) == (1, 0)
    assert s.escalation_rate == pytest.approx(1 / 7)
    assert s.routed_up_rate == pytest.approx(1 / 7)
    assert s.routing_share["local-llama"] == pytest.approx(5 / 7)
    tier1 = next(t for t in s.tiers if t.tier == 1)
    assert (tier1.requests, tier1.truncated) == (5, 1) and tier1.truncation_rate == pytest.approx(0.2)
    assert tier1.p50_latency_ms == pytest.approx(1000)


def test_headline_always_pairs_savings_with_quality(db: Database) -> None:
    s = summarise(load_frames(db), tier3_model=PRO.name, tier3_pricing=PRO.pricing)
    headline = s.headline()
    assert "quality parity" in headline and "net of verification" in headline and "n=2" in headline
    assert "small sample" in headline
    assert headline.startswith("Reduced LLM cost by 30% net of verification (75% before verification)")
    worse = s.model_copy(update={"net_savings_pct": -0.6})
    assert worse.headline().startswith("Increased LLM cost by 60% net of verification")


def test_no_quality_means_no_savings_claim(tmp_path: Path) -> None:
    database = Database(tmp_path / "q.db")
    Repository(database).insert_request(request("x", serve_cost_usd=0, input_tokens=10, output_tokens=10,
                                                verification_status="not_sampled"))
    s = summarise(load_frames(database), tier3_model=PRO.name, tier3_pricing=PRO.pricing)
    assert s.quality.parity is None
    assert "no savings claim" in s.headline() and "Reduced" not in s.headline()


def test_filters(db: Database) -> None:
    repo = Repository(db)
    repo.insert_request(request("old", created_at=utc_now() - timedelta(days=10), mode="evaluation",
                                verification_status="not_sampled"))
    assert len(load_frames(db, mode="evaluation").requests) == 1
    assert len(load_frames(db, mode="normal").requests) == 8
    recent = load_frames(db, since=utc_now() - timedelta(days=1))
    assert "old" not in set(recent.requests["id"])
    assert len(recent.verifications) == 4 and len(recent.escalations) == 1


def test_chart_series(db: Database) -> None:
    frames = load_frames(db)
    daily = daily_costs(frames, tier3_model=PRO.name, tier3_pricing=PRO.pricing)
    assert len(daily) == 1
    assert daily["actual_cost_usd"].iloc[0] == pytest.approx(0.0215)
    esc = daily_escalations(frames)
    assert esc["escalations"].tolist() == [1] and esc["requests"].tolist() == [7]
    assert quality_distribution(frames)["count"].tolist() == [0, 1, 0, 0, 1]  # sampled only
    dist = routing_distribution(frames)
    assert dist["requests"].sum() == 7 and dist["share"].sum() == pytest.approx(1)


@pytest.mark.parametrize(("k", "n"), [(0, 10), (5, 10), (10, 10), (45, 50)])
def test_wilson_interval_brackets_the_rate(k: int, n: int) -> None:
    low, high = wilson_interval(k, n)
    assert 0 <= low <= k / n <= high <= 1
    assert wilson_interval(0, 0) == (0.0, 1.0)
