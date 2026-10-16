"""Fill the README headline and results table from data/loadtest_report.json (no hand-copied numbers).

    python scripts/fill_readme.py
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from autopilot.settings import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
REPORT = PROJECT_ROOT / "data" / "loadtest_report.json"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def pct(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


def headline(report: dict[str, Any]) -> str:
    c, q, run = report["costs"], report["quality"], report["run"]
    return (
        f"Cut LLM cost by {pct(c['gross_savings_pct'])} vs. sending everything to the top model "
        f"({pct(c['projection']['net_savings_pct'])} net after verifying a random "
        f"{pct(c['projection']['sample_rate'])} of answers), while keeping {pct(q['parity'])} quality parity "
        f"(95% CI {pct(q['ci_low'])}–{pct(q['ci_high'])}) on {run['ok']:,} requests."
    )


def results_table(report: dict[str, Any]) -> str:
    c, q, r = report["costs"], report["quality"], report["routing"]
    share = " · ".join(f"{m} {pct(s)}" for m, s in sorted(r["distribution"].items(), key=lambda kv: -kv[1]))
    tiers = report["per_tier"]
    latency = " · ".join(
        f"T{t['tier']} {t['p50_latency_ms'] / 1000:.1f}s / {t['p95_latency_ms'] / 1000:.1f}s" for t in tiers
    )
    truncation = " · ".join(f"T{t['tier']} {pct(t['truncation_rate'], 1)}" for t in tiers)
    excluded = (q["excluded_routed_truncated"] + q["excluded_reference_truncated"]
                + q["excluded_judge_unparseable"] + q["verification_failed_jobs"] + q["pending"])
    rows = [
        ("Requests (held-out Dolly pool)", f"{report['run']['ok']:,} answered, {report['run']['failed']} failed"),
        ("Routing", share),
        ("Everything on Tier 3 (baseline)", f"${c['all_tier3_baseline_usd']:.2f} " + (
            "(all measured: real Tier-3 cost for every request)" if c["baseline_measured_share"] >= 0.999
            else f"({pct(c['baseline_measured_share'])} measured, rest estimated from tokens)")),
        ("Autopilot answering cost", f"${c['serve_usd']:.2f} → **gross savings {pct(c['gross_savings_pct'])}**"),
        ("Quality parity (judge: as good as Tier 3)", f"**{pct(q['parity'], 1)}** — 95% CI {pct(q['ci_low'], 1)}–{pct(q['ci_high'], 1)}, n={q['judged']}; {excluded} excluded and reported"),
        ("Verification, every downgraded answer (load test)", f"${c['verification_usd']:.2f} → net savings in this run {pct(c['net_savings_pct'])}"),
        ("Projected net savings, 15% verified (production)", f"**{pct(c['projection']['net_savings_pct'])}**"),
        ("Escalation rate (failures logged)", pct(r["escalation_rate"], 1)),
        ("Routed up for low confidence", pct(r["routed_up_rate"], 1)),
        ("Latency P50 / P95 by tier", latency),
        ("Truncated at max_tokens (excluded from parity)", truncation),
        ("Load-test bill", f"${c['total_usd']:.2f} at peak prices · ${c['offpeak_equivalent_total_usd']:.2f} at off-peak prices"),
    ]
    lines = [START, "", "| Load test | Result |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows], "",
             "Full report: [`data/loadtest_report.md`](data/loadtest_report.md) · raw numbers: "
             "[`data/loadtest_report.json`](data/loadtest_report.json)", "", END]
    return "\n".join(lines)


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    text = re.sub(r"^\*\*.*\*\*$", f"**{headline(report)}**", text, count=1, flags=re.MULTILINE)
    table = results_table(report)
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: table, text, count=1, flags=re.DOTALL)
    else:
        text = text.replace("{{RESULTS_TABLE}}", table)
    README.write_text(text, encoding="utf-8")
    print(headline(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
