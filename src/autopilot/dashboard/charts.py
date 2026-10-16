"""Altair chart builders for the dashboard (pure functions → easy to test).

Colors: the validated reference palette (slots 1-2 pass every colorblind and contrast check
in both modes), with separate dark-mode steps. One y-axis per chart, thin marks, hairline
grid, tooltips on every mark; every chart also has a data table next to it in the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import pandas as pd

from autopilot.analytics.metrics import TierRow


@dataclass(frozen=True)
class Palette:
    series_1: str
    series_2: str
    surface: str
    ink: str
    muted: str
    grid: str
    baseline: str


LIGHT = Palette("#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7")
DARK = Palette("#3987e5", "#d95926", "#1a1a19", "#f0efec", "#898781", "#2c2c2a", "#383835")

FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def palette_for(theme: str | None) -> Palette:
    return DARK if theme == "dark" else LIGHT


def _style(chart: alt.Chart | alt.LayerChart, p: Palette, height: int = 240) -> alt.Chart | alt.LayerChart:
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=p.grid, gridWidth=1, domainColor=p.baseline, tickColor=p.baseline,
            labelColor=p.muted, titleColor=p.muted, labelFont=FONT, titleFont=FONT, titleFontWeight="normal",
        )
        .configure_legend(labelColor=p.ink, titleColor=p.muted, labelFont=FONT, titleFont=FONT, orient="top")
    )


def _two_series(p: Palette, names: list[str]) -> alt.Scale:
    return alt.Scale(domain=names, range=[p.series_1, p.series_2])


def cost_per_day(daily: pd.DataFrame, p: Palette) -> alt.Chart:
    """Stacked bars: what we actually paid each day, split into answering vs verifying."""
    long = daily.melt(
        id_vars="day", value_vars=["serve_cost_usd", "verification_cost_usd"], var_name="part", value_name="usd"
    ).replace({"part": {"serve_cost_usd": "Answering", "verification_cost_usd": "Verification"}})
    long["date"] = pd.to_datetime(long["day"]).dt.strftime("%b %d")
    chart = alt.Chart(long).mark_bar(stroke=p.surface, strokeWidth=2, size=28).encode(
        x=alt.X("date:O", title=None, sort=alt.SortField("day"), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("sum(usd):Q", title="USD", axis=alt.Axis(format="$.3f")),
        color=alt.Color("part:N", title=None, scale=_two_series(p, ["Answering", "Verification"])),
        order=alt.Order("part:N"),
        tooltip=[alt.Tooltip("date:O", title="Day"), alt.Tooltip("part:N", title="Part"),
                 alt.Tooltip("sum(usd):Q", title="USD", format="$.4f")],
    )
    return _style(chart, p)


def actual_vs_baseline(daily: pd.DataFrame, p: Palette) -> alt.Chart:
    """Cumulative cost: autopilot (incl. verification) vs sending everything to Tier 3."""
    frame = daily.sort_values("day").assign(
        actual=lambda d: d["actual_cost_usd"].cumsum(), baseline=lambda d: d["baseline_cost_usd"].cumsum()
    )
    long = frame.melt(id_vars="day", value_vars=["actual", "baseline"], var_name="series", value_name="usd").replace(
        {"series": {"actual": "Autopilot (incl. verification)", "baseline": "Everything on Tier 3"}}
    )
    names = ["Autopilot (incl. verification)", "Everything on Tier 3"]
    base = alt.Chart(long).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("usd:Q", title="Cumulative USD", axis=alt.Axis(format="$.3f")),
        color=alt.Color("series:N", title=None, scale=_two_series(p, names)),
        tooltip=[alt.Tooltip("day:T", title="Day"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("usd:Q", title="Cumulative USD", format="$.4f")],
    )
    return _style(base.mark_line(strokeWidth=2) + base.mark_point(size=64, filled=True, stroke=p.surface, strokeWidth=2), p)


def routing_share(dist: pd.DataFrame, p: Palette) -> alt.Chart:
    """Share of requests answered by each model (one series → one color)."""
    frame = dist.assign(label=lambda d: d["model"] + " (tier " + d["tier"].astype(str) + ")")
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", title=None, sort=alt.SortField("tier")),
        x=alt.X("share:Q", title="Share of requests", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("requests:Q"), alt.Tooltip("share:Q", format=".0%")],
    )
    bars = base.mark_bar(color=p.series_1, cornerRadiusEnd=4, size=18)
    labels = base.mark_text(align="left", dx=6, color=p.ink, font=FONT).encode(text=alt.Text("share:Q", format=".0%"))
    return _style(bars + labels, p, height=60 + 36 * len(frame))


def quality_histogram(dist: pd.DataFrame, threshold: int, p: Palette) -> alt.LayerChart:
    """How the judge scored sampled downgraded answers (5 = as good as Tier 3)."""
    base = alt.Chart(dist).encode(
        x=alt.X("agreement:O", axis=alt.Axis(labelAngle=0),
                title=f"Judge agreement score (5 = as good as Tier 3) · line = pass threshold ≥ {threshold}"),
        y=alt.Y("count:Q", title="Verified requests", axis=alt.Axis(tickMinStep=1, format="d")),
        tooltip=[alt.Tooltip("agreement:O", title="Score"), alt.Tooltip("count:Q", title="Requests")],
    )
    bars = base.mark_bar(color=p.series_1, cornerRadiusEnd=4, size=36)
    rule = alt.Chart(pd.DataFrame({"agreement": [threshold]})).mark_rule(
        color=p.muted, strokeWidth=1, xOffset=-24
    ).encode(x="agreement:O")
    return _style(bars + rule, p)


def escalation_rate(daily: pd.DataFrame, p: Palette) -> alt.Chart:
    """Share of requests with an escalation (sync replacements + async logged failures) per day."""
    base = alt.Chart(daily).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("escalation_rate:Q", title="Escalation rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("day:T", title="Day"), alt.Tooltip("requests:Q"), alt.Tooltip("escalations:Q"),
                 alt.Tooltip("escalation_rate:Q", title="Rate", format=".1%")],
    )
    line = base.mark_line(color=p.series_1, strokeWidth=2)
    points = base.mark_point(color=p.series_1, size=64, filled=True, stroke=p.surface, strokeWidth=2)
    return _style(line + points, p)


def latency_by_tier(tiers: list[TierRow], p: Palette) -> alt.Chart:
    """End-to-end latency percentiles per served tier."""
    rows = [
        {"tier": f"Tier {t.tier}", "stat": stat, "seconds": value / 1000}
        for t in tiers
        for stat, value in (("P50", t.p50_latency_ms), ("P95", t.p95_latency_ms))
        if value is not None
    ]
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4, size=16, stroke=p.surface, strokeWidth=2).encode(
        x=alt.X("tier:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("stat:N"),
        y=alt.Y("seconds:Q", title="Seconds"),
        color=alt.Color("stat:N", title=None, scale=_two_series(p, ["P50", "P95"])),
        tooltip=[alt.Tooltip("tier:N"), alt.Tooltip("stat:N"), alt.Tooltip("seconds:Q", format=".1f")],
    )
    return _style(chart, p)


def truncation_by_tier(tiers: list[TierRow], p: Palette) -> alt.Chart:
    """Share of answers cut off at max_tokens, per served tier (these are excluded from parity)."""
    frame = pd.DataFrame(
        [{"tier": f"Tier {t.tier}", "rate": t.truncation_rate, "truncated": t.truncated, "requests": t.requests}
         for t in tiers]
    )
    base = alt.Chart(frame).encode(
        x=alt.X("tier:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Truncated (hit max_tokens)", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, max(0.1, float(frame["rate"].max()) * 1.2)])),
        tooltip=[alt.Tooltip("tier:N"), alt.Tooltip("truncated:Q"), alt.Tooltip("requests:Q"),
                 alt.Tooltip("rate:Q", format=".1%")],
    )
    bars = base.mark_bar(color=p.series_1, cornerRadiusEnd=4, size=36)
    labels = base.mark_text(dy=-8, color=p.ink, font=FONT).encode(text=alt.Text("rate:Q", format=".0%"))
    return _style(bars + labels, p)
