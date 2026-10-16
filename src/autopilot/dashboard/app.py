"""LLM Cost Autopilot dashboard.

    streamlit run src/autopilot/dashboard/app.py

Honest-metrics rule: savings are rendered in exactly one place, :func:`honest_panel`, which
always shows them next to quality parity and verification overhead — and hides them until
at least one random-sample verification exists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from autopilot.analytics.metrics import (
    MIN_RELIABLE_SAMPLE,
    Frames,
    HonestSummary,
    daily_costs,
    daily_escalations,
    load_frames,
    quality_distribution,
    routing_distribution,
    summarise,
)
from autopilot.dashboard import charts
from autopilot.llm.catalog import load_catalog
from autopilot.routing.config import load_routing_config
from autopilot.settings import load_settings
from autopilot.storage.db import Database
from autopilot.verification.rules import load_quality_config

PERIODS = {"All time": None, "Last 24 hours": timedelta(days=1), "Last 7 days": timedelta(days=7),
           "Last 30 days": timedelta(days=30)}
MODES = {"All traffic": None, "Normal": "normal", "Evaluation": "evaluation"}


def pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.0%}" if signed else f"{value:.0%}"


def usd(value: float) -> str:
    return f"${value:,.4f}" if abs(value) < 1 else f"${value:,.2f}"


def usd_change(value: float) -> str:
    """Sign first ("-$1.21", "+$1.17") so Streamlit picks the right arrow and colour."""
    return ("-" if value < 0 else "+") + usd(abs(value))


def escape_dollars(text: str) -> str:
    """Streamlit markdown treats text between two $ signs as LaTeX; escape them."""
    return text.replace("$", "\\$")


@st.cache_data(ttl=15, show_spinner=False)
def cached_frames(db_path: str, mode: str | None, since: datetime | None) -> Frames:
    return load_frames(Database(Path(db_path)), mode=mode, since=since)


def honest_panel(summary: HonestSummary) -> None:
    """The ONLY place savings appear — always beside quality parity and verification overhead."""
    q = summary.quality
    st.markdown(f"**{summary.headline()}**")
    cols = st.columns(4)
    if q.parity is None:
        cols[0].metric("Net savings (after verification)", "—", help="Hidden until quality has been measured.")
        cols[1].metric("Quality parity", "Not measured", help="No randomly sampled verification yet.")
        cols[2].metric("Verification overhead", usd(summary.verification_cost_usd))
        cols[3].metric("Gross savings (before verification)", "—", help="Hidden until quality has been measured.")
    else:
        cols[0].metric("Net savings (after verification)", pct(summary.net_savings_pct),
                       delta=usd_change(summary.net_savings_usd), delta_color="normal")
        cols[1].metric("Quality parity", pct(q.parity),
                       delta=f"95% CI {pct(q.ci_low)}–{pct(q.ci_high)} · n={q.judged}", delta_color="off",
                       delta_arrow="off",
                       help="Share of randomly sampled downgraded answers the judge rated as good as Tier 3.")
        cols[2].metric("Verification overhead", usd(summary.verification_cost_usd),
                       delta=f"{pct(summary.verification_overhead_pct)} of baseline", delta_color="off",
                       delta_arrow="off")
        cols[3].metric("Gross savings (before verification)", pct(summary.gross_savings_pct),
                       delta=usd_change(summary.gross_savings_usd), delta_color="normal")
    if q.parity is not None and not q.reliable:
        st.warning(f"Small sample: quality parity rests on {q.judged} judged answers (< {MIN_RELIABLE_SAMPLE}). "
                   "Read the confidence interval, not the point estimate.")
    st.caption(escape_dollars(
        f"Excluded from parity — truncated routed answers: {q.excluded_routed_truncated} · truncated references: "
        f"{q.excluded_reference_truncated} · unparseable judge replies: {q.excluded_judge_unparseable} · "
        f"pending: {q.pending} · failed jobs: {q.verification_failed_jobs}. "
        f"Sync low-confidence checks (not in parity): {q.sync_passed}/{q.sync_judged} passed. "
        f"Baseline measured for {pct(summary.baseline_measured_share)} of requests, estimated from tokens for the rest. "
        f"Actual cost {usd(summary.total_cost_usd)} vs all-Tier-3 {usd(summary.baseline_cost_usd)}."
    ))


def operations_row(summary: HonestSummary) -> None:
    cols = st.columns(5)
    cols[0].metric("Requests", f"{summary.requests:,}")
    cols[1].metric("Errors", f"{summary.errors:,}")
    cols[2].metric("Escalation rate", pct(summary.escalation_rate),
                   help="Sync replacements + async logged failures, per request. Async failures are only found "
                        "in the verified sample.")
    cols[3].metric("Routed up (low confidence)", pct(summary.routed_up_rate))
    cols[4].metric("Failover (model down)", pct(summary.failover_rate))


def chart_card(title: str, chart: object, data: pd.DataFrame, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)
    st.altair_chart(chart, width="stretch", theme=None)
    with st.expander("Data"):
        st.dataframe(data, hide_index=True, width="stretch")


def render_dashboard(db_path: Path, *, mode: str | None = None, since: datetime | None = None) -> None:
    """Honest panel + operations row + all charts for one database (reused by the public demo)."""
    catalog = load_catalog()
    routing = load_routing_config(catalog)
    tier3 = catalog.get(routing.tiers[3])
    threshold = load_quality_config(catalog).default.pass_threshold

    if not db_path.exists():
        st.info("No database yet — send some requests first (`make ask`).")
        return
    frames = cached_frames(str(db_path), mode, since)
    if frames.requests.empty:
        st.info("No requests in this period yet — send some with `make ask`.")
        return

    summary = summarise(frames, tier3_model=tier3.name, tier3_pricing=tier3.pricing)
    honest_panel(summary)
    operations_row(summary)
    p = charts.palette_for(st.context.theme.type)

    daily = daily_costs(frames, tier3_model=tier3.name, tier3_pricing=tier3.pricing)
    left, right = st.columns(2)
    with left:
        chart_card("Cost per day", charts.cost_per_day(daily, p), daily,
                   "What was actually paid: answering + verification.")
    with right:
        chart_card("Actual vs everything on Tier 3", charts.actual_vs_baseline(daily, p), daily,
                   "Cumulative. The gap is net savings; verification cost is included in the actual line.")

    left, right = st.columns(2)
    with left:
        dist = routing_distribution(frames)
        chart_card("Routing distribution", charts.routing_share(dist, p), dist)
    with right:
        quality = quality_distribution(frames)
        chart_card("Quality score distribution", charts.quality_histogram(quality, threshold, p), quality,
                   "Random-sample verifications only. Scores below the line are routing failures.")

    left, right = st.columns(2)
    with left:
        escalations = daily_escalations(frames)
        chart_card("Escalation rate over time", charts.escalation_rate(escalations, p), escalations)
    with right:
        tier_frame = pd.DataFrame([t.model_dump() for t in summary.tiers])
        chart_card("Latency by tier", charts.latency_by_tier(summary.tiers, p), tier_frame,
                   "End-to-end, including classification and any sync verification.")

    chart_card("Truncation by tier", charts.truncation_by_tier(summary.tiers, p), tier_frame,
               "Answers cut off at max_tokens are excluded from quality parity — shown here so nothing is hidden.")


def main() -> None:
    st.set_page_config(page_title="LLM Cost Autopilot", page_icon="🧭", layout="wide")
    db_path = load_settings().database.resolved_path()
    st.title("LLM Cost Autopilot")
    filters = st.columns([1, 1, 2])
    mode = MODES[filters[0].selectbox("Traffic", list(MODES))]
    period = PERIODS[filters[1].selectbox("Period", list(PERIODS))]
    filters[2].caption(f"Database: `{db_path}` · costs at DeepSeek peak prices (off-peak is 50% cheaper)")
    since = datetime.now(UTC) - period if period else None
    render_dashboard(db_path, mode=mode, since=since)


if __name__ == "__main__":
    main()
