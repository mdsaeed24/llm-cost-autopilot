"""Every number the project reports, defined once. Used by the dashboard, /v1/stats and the load test.

Honesty rules baked in here:
- **Savings never travel alone.** :class:`HonestSummary` carries savings, quality parity,
  verification overhead and net savings together, and :meth:`HonestSummary.headline`
  refuses to state savings without a quality number.
- **Verification is a cost.** Net savings subtract every Tier-3 reference and judge call.
- **Quality parity uses only the random sample** (``source = 'sampled'``). Sync checks only
  look at low-confidence requests, so including them would bias the number.
- **Exclusions are counted, never hidden**: truncated routed answers, truncated references,
  and unparseable judge replies are reported next to parity, per tier where relevant.
- **The all-Tier-3 baseline is measured where possible** (actual Tier-3 cost for requests
  served by, or verified against, Tier 3) and estimated from token counts otherwise; the
  measured share is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from autopilot.llm.schemas import ModelPricing
from autopilot.storage.db import Database, iso

MIN_RELIABLE_SAMPLE = 30


# ---------- loading ----------

@dataclass
class Frames:
    """The three tables the metrics need, already filtered to one slice of traffic."""

    requests: pd.DataFrame
    verifications: pd.DataFrame
    escalations: pd.DataFrame


def _filter(prefix: str, mode: str | None, since: datetime | None) -> tuple[str, list[object]]:
    """WHERE clause on the requests table (aliased ``prefix``) for one slice of traffic."""
    clauses, params = ["1 = 1"], []
    if mode:
        clauses.append(f"{prefix}.mode = ?")
        params.append(mode)
    if since:
        clauses.append(f"{prefix}.created_at >= ?")
        params.append(iso(since))
    return " AND ".join(clauses), params


def load_frames(db: Database, *, mode: str | None = None, since: datetime | None = None) -> Frames:
    """Read requests (optionally one mode / since a time) plus their verifications and escalations."""
    clause, params = _filter("r", mode, since)
    with db.connect() as conn:
        requests = pd.read_sql_query(f"SELECT r.* FROM requests r WHERE {clause}", conn, params=params)
        verifications = pd.read_sql_query(
            f"SELECT v.* FROM verifications v JOIN requests r ON r.id = v.request_id WHERE {clause}", conn, params=params
        )
        escalations = pd.read_sql_query(
            f"SELECT e.* FROM escalations e JOIN requests r ON r.id = e.request_id WHERE {clause}", conn, params=params
        )
    for frame in (requests, verifications, escalations):
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return Frames(requests, verifications, escalations)


# ---------- baseline ----------

def with_baseline(frames: Frames, tier3_model: str, tier3_pricing: ModelPricing) -> pd.DataFrame:
    """Successful requests with ``baseline_cost_usd`` (what Tier 3 would have cost) and ``baseline_measured``."""
    ok = frames.requests[frames.requests["status"] == "ok"].copy()
    reference_cost = frames.verifications.set_index("request_id")["reference_cost_usd"] if len(frames.verifications) else pd.Series(dtype=float)
    estimate = (
        ok["input_tokens"].fillna(0) * tier3_pricing.input_cache_miss_per_1m
        + ok["output_tokens"].fillna(0) * tier3_pricing.output_per_1m
    ) / 1_000_000
    served_by_tier3 = ok["served_model"] == tier3_model
    verified_cost = ok["id"].map(reference_cost)
    ok["baseline_cost_usd"] = np.where(
        served_by_tier3, ok["serve_cost_usd"], np.where(verified_cost.notna(), verified_cost, estimate)
    )
    ok["baseline_measured"] = served_by_tier3 | verified_cost.notna()
    return ok


# ---------- summary ----------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion. Honest for small samples, unlike ±2·SE."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class QualityParity(BaseModel):
    """Share of randomly sampled, downgraded answers the judge rated as good as Tier 3."""

    model_config = ConfigDict(extra="forbid")

    judged: int
    passed: int
    parity: float | None
    ci_low: float | None
    ci_high: float | None
    reliable: bool  # judged >= MIN_RELIABLE_SAMPLE
    excluded_routed_truncated: int
    excluded_reference_truncated: int
    excluded_judge_unparseable: int
    verification_failed_jobs: int
    pending: int
    sync_judged: int  # low-confidence checks; reported separately, not in parity
    sync_passed: int


class TierRow(BaseModel):
    """Per-served-tier operational numbers."""

    model_config = ConfigDict(extra="forbid")

    tier: int
    requests: int
    truncated: int
    truncation_rate: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None


class HonestSummary(BaseModel):
    """Savings + quality + overhead, always together."""

    model_config = ConfigDict(extra="forbid")

    requests: int
    errors: int
    serve_cost_usd: float
    verification_cost_usd: float
    total_cost_usd: float
    baseline_cost_usd: float
    baseline_measured_share: float
    gross_savings_usd: float
    gross_savings_pct: float | None
    net_savings_usd: float
    net_savings_pct: float | None
    verification_overhead_pct: float | None  # verification cost as % of baseline
    quality: QualityParity
    escalations_async: int
    escalations_sync: int
    escalation_rate: float | None
    routed_up_rate: float | None
    failover_rate: float | None
    routing_share: dict[str, float]
    tiers: list[TierRow]

    def headline(self) -> str:
        """The one-line claim. Never states savings without quality parity."""
        q = self.quality
        if q.parity is None or self.net_savings_pct is None:
            return (
                f"{self.requests} requests — quality not measured yet, so no savings claim is made "
                f"(gross cost change would be {_pct(self.gross_savings_pct)} before verification)."
            )
        caution = "" if q.reliable else f" (small sample: n={q.judged} < {MIN_RELIABLE_SAMPLE})"
        verb = "Reduced" if self.net_savings_pct >= 0 else "Increased"
        return (
            f"{verb} LLM cost by {_pct(abs(self.net_savings_pct))} net of verification "
            f"({_pct(self.gross_savings_pct)} before verification) while keeping {_pct(q.parity)} quality parity "
            f"[95% CI {_pct(q.ci_low)}–{_pct(q.ci_high)}, n={q.judged}] on {self.requests} requests{caution}."
        )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarise(frames: Frames, *, tier3_model: str, tier3_pricing: ModelPricing) -> HonestSummary:
    """Compute :class:`HonestSummary` for one slice of traffic."""
    requests = frames.requests
    ok = with_baseline(frames, tier3_model, tier3_pricing)
    serve = float(ok["serve_cost_usd"].sum())
    verification = float(requests["verification_cost_usd"].sum())  # includes checks on any request
    baseline = float(ok["baseline_cost_usd"].sum())
    total = serve + verification

    v = frames.verifications
    sampled = v[v["source"] == "sampled"] if len(v) else v
    judged = sampled[sampled["passed"].notna()] if len(sampled) else sampled
    passed = int((judged["passed"] == 1).sum()) if len(judged) else 0
    n_judged = len(judged)
    low, high = wilson_interval(passed, n_judged) if n_judged else (None, None)
    sync = v[v["source"] == "sync"] if len(v) else v
    sync_judged = sync[sync["passed"].notna()] if len(sync) else sync

    def excluded(reason: str) -> int:
        return int((sampled["excluded_reason"] == reason).sum()) if len(sampled) else 0

    quality = QualityParity(
        judged=n_judged,
        passed=passed,
        parity=_ratio(passed, n_judged),
        ci_low=low,
        ci_high=high,
        reliable=n_judged >= MIN_RELIABLE_SAMPLE,
        excluded_routed_truncated=int((requests["verification_status"] == "excluded_truncated").sum()),
        excluded_reference_truncated=excluded("reference_truncated"),
        excluded_judge_unparseable=excluded("judge_unparseable"),
        verification_failed_jobs=int((requests["verification_status"] == "failed").sum()),
        pending=int((requests["verification_status"] == "queued").sum()),
        sync_judged=len(sync_judged),
        sync_passed=int((sync_judged["passed"] == 1).sum()) if len(sync_judged) else 0,
    )

    e = frames.escalations
    n_ok = len(ok)
    return HonestSummary(
        requests=len(requests),
        errors=int((requests["status"] == "error").sum()),
        serve_cost_usd=serve,
        verification_cost_usd=verification,
        total_cost_usd=total,
        baseline_cost_usd=baseline,
        baseline_measured_share=float(ok["baseline_measured"].mean()) if n_ok else 0.0,
        gross_savings_usd=baseline - serve,
        gross_savings_pct=_ratio(baseline - serve, baseline),
        net_savings_usd=baseline - total,
        net_savings_pct=_ratio(baseline - total, baseline),
        verification_overhead_pct=_ratio(verification, baseline),
        quality=quality,
        escalations_async=int((e["mode"] == "async").sum()) if len(e) else 0,
        escalations_sync=int((e["mode"] == "sync").sum()) if len(e) else 0,
        escalation_rate=_ratio(len(e), n_ok),
        routed_up_rate=_ratio(int(ok["routed_up"].sum()), n_ok),
        failover_rate=_ratio(int((ok["served_tier"] != ok["routed_tier"]).sum()), n_ok),
        routing_share={str(k): float(v) for k, v in ok["served_model"].value_counts(normalize=True).items()},
        tiers=tier_table(ok),
    )


def tier_table(ok: pd.DataFrame) -> list[TierRow]:
    """Truncation rate and latency percentiles per served tier."""
    rows: list[TierRow] = []
    for tier, group in ok.groupby("served_tier"):
        latency = group["latency_ms"].dropna()
        rows.append(
            TierRow(
                tier=int(tier),
                requests=len(group),
                truncated=int(group["truncated"].sum()),
                truncation_rate=float(group["truncated"].mean()),
                p50_latency_ms=float(latency.quantile(0.5)) if len(latency) else None,
                p95_latency_ms=float(latency.quantile(0.95)) if len(latency) else None,
            )
        )
    return rows


# ---------- time series & distributions (for charts) ----------

def daily_costs(frames: Frames, *, tier3_model: str, tier3_pricing: ModelPricing) -> pd.DataFrame:
    """Per day: serve cost, verification cost, actual total, and the all-Tier-3 baseline."""
    ok = with_baseline(frames, tier3_model, tier3_pricing)
    columns = ["day", "serve_cost_usd", "verification_cost_usd", "actual_cost_usd", "baseline_cost_usd"]
    if ok.empty:
        return pd.DataFrame(columns=columns)
    ok["day"] = ok["created_at"].dt.floor("D")
    daily = ok.groupby("day")[["serve_cost_usd", "verification_cost_usd", "baseline_cost_usd"]].sum().reset_index()
    daily["actual_cost_usd"] = daily["serve_cost_usd"] + daily["verification_cost_usd"]
    return daily[columns]


def daily_escalations(frames: Frames) -> pd.DataFrame:
    """Per day: requests, escalations (sync + async) and the escalation rate."""
    ok = frames.requests[frames.requests["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame(columns=["day", "requests", "escalations", "escalation_rate"])
    per_day = ok.assign(day=ok["created_at"].dt.floor("D")).groupby("day").size().rename("requests")
    e = frames.escalations
    esc = (e.assign(day=e["created_at"].dt.floor("D")).groupby("day").size() if len(e) else pd.Series(dtype=int))
    out = pd.concat([per_day, esc.rename("escalations")], axis=1).fillna(0).reset_index().rename(columns={"index": "day"})
    out["escalations"] = out["escalations"].astype(int)
    out["escalation_rate"] = out["escalations"] / out["requests"]
    return out


def quality_distribution(frames: Frames) -> pd.DataFrame:
    """Count of sampled verifications at each agreement score 1-5 (zeros included)."""
    v = frames.verifications
    sampled = v[(v["source"] == "sampled") & v["agreement"].notna()] if len(v) else v
    counts = sampled["agreement"].astype(int).value_counts() if len(sampled) else pd.Series(dtype=int)
    return pd.DataFrame({"agreement": list(range(1, 6)), "count": [int(counts.get(s, 0)) for s in range(1, 6)]})


def routing_distribution(frames: Frames) -> pd.DataFrame:
    """Requests per served model, with share."""
    ok = frames.requests[frames.requests["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame(columns=["model", "tier", "requests", "share"])
    grouped = ok.groupby(["served_model", "served_tier"]).size().reset_index(name="requests")
    grouped["share"] = grouped["requests"] / grouped["requests"].sum()
    return grouped.rename(columns={"served_model": "model", "served_tier": "tier"}).sort_values("tier")
