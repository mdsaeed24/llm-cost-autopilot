"""README filler: headline + results table come from the load-test report, and refills are idempotent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import load_script

REPORT = {
    "run": {"ok": 998, "failed": 2},
    "costs": {"gross_savings_pct": 0.62, "net_savings_pct": -0.8, "serve_usd": 0.5, "verification_usd": 3.0,
              "total_usd": 3.5, "all_tier3_baseline_usd": 1.3, "baseline_measured_share": 0.9,
              "offpeak_equivalent_total_usd": 1.75, "projection": {"sample_rate": 0.15, "net_savings_pct": 0.27}},
    "quality": {"parity": 0.81, "ci_low": 0.78, "ci_high": 0.84, "judged": 740, "excluded_routed_truncated": 3,
                "excluded_reference_truncated": 4, "excluded_judge_unparseable": 0, "verification_failed_jobs": 0,
                "pending": 0},
    "routing": {"distribution": {"local-llama": 0.45, "deepseek-flash": 0.33, "deepseek-v4-pro": 0.22},
                "escalation_rate": 0.14, "routed_up_rate": 0.2},
    "per_tier": [{"tier": 1, "p50_latency_ms": 6000, "p95_latency_ms": 12000, "truncation_rate": 0.004},
                 {"tier": 3, "p50_latency_ms": 9000, "p95_latency_ms": 30000, "truncation_rate": 0.01}],
}


def test_fill_readme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script("fill_readme")
    readme, report = tmp_path / "README.md", tmp_path / "report.json"
    readme.write_text("# Title\n\n**{{HEADLINE}}**\n\nIntro.\n\n{{RESULTS_TABLE}}\n\nRest.\n")
    report.write_text(json.dumps(REPORT))
    monkeypatch.setattr(script, "README", readme)
    monkeypatch.setattr(script, "REPORT", report)

    assert script.main() == 0
    text = readme.read_text()
    assert "**Cut LLM cost by 62% vs. sending everything to the top model (27% net after verifying a random 15%" in text
    assert "81% quality parity (95% CI 78%–84%) on 998 requests" in text
    assert "| Projected net savings, 15% verified (production) | **27%** |" in text
    assert "net savings in this run -80%" in text and "7 excluded and reported" in text
    assert "{{" not in text

    script.main()  # refilling replaces the block instead of duplicating it
    assert readme.read_text().count("<!-- results:start -->") == 1
