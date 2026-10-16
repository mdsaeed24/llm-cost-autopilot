"""Public demo page: renders from the committed snapshots, routes example prompts, never calls an LLM."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from autopilot.analytics.metrics import load_frames, summarise
from autopilot.dashboard import showcase
from autopilot.llm.catalog import load_catalog
from autopilot.settings import PROJECT_ROOT
from autopilot.storage.db import Database

APP = str(PROJECT_ROOT / "streamlit_app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_page_layout(app: AppTest) -> None:
    assert app.title[0].value == "🧭 LLM Cost Autopilot"
    assert "quality parity" in app.success[0].value and "net after verifying" in app.success[0].value
    assert [t.label for t in app.tabs] == [
        "🧭 Try the router", "📊 Load-test dashboard", "🔁 Before vs after the fix", "🏗 How it works",
    ]
    assert any("No LLM is called" in c.value for c in app.caption)


@pytest.mark.parametrize(("example", "tiers"), [
    ("Extract from text", {1, 2}),
    ("Fact from memory", {2, 3}),       # rules-v3: memory facts no longer go to the 3B model
    ("Reasoning puzzle", {2, 3}),
    ("Creative", {2, 3}),
])
def test_router_examples(example: str, tiers: set[int]) -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio[0].set_value(example).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    predicted = int(metrics["Predicted tier"].split()[-1])
    assert predicted in tiers, (example, metrics)
    assert metrics["Routed to"] in {"local-llama", "deepseek-flash", "deepseek-v4-pro"}


def test_dashboard_tab_switches_between_real_runs() -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    parity = lambda: next(m.value for m in at.metric if m.label == "Quality parity")  # noqa: E731
    assert parity() == "85%"  # run 2 (v3) by default
    at.selectbox[0].set_value(list(showcase.RUNS)[1]).run()
    assert parity() == "78%"  # run 1 (v2)


@pytest.mark.parametrize(("db", "requests", "judged"), [("run1_v2_1000.db", 1000, 751), ("run2_v3_300.db", 300, 214)])
def test_snapshots_match_the_reports(db: str, requests: int, judged: int) -> None:
    pro = load_catalog().get("deepseek-v4-pro")
    summary = summarise(load_frames(Database(PROJECT_ROOT / "demo" / db)), tier3_model=pro.name, tier3_pricing=pro.pricing)
    assert (summary.requests, summary.quality.judged) == (requests, judged)


def test_comparison_table_and_cost_estimate() -> None:
    table = showcase.comparison_table(showcase.load_json(showcase.DATA / "v2_vs_v3_comparison.json"))
    assert list(table.columns) == ["Same 300 prompts", "v2 (before)", "v3 (after)"]
    assert table.iloc[1].tolist()[1:] == ["76.7% (71%–82%)", "85.0% (80%–89%)"]

    router = showcase.get_router()
    decision = router.decide("Write a sonnet about tea and explain your choice of metre.")
    routed, top = showcase.estimate_costs("x" * 400, decision, router.catalog, "deepseek-v4-pro")
    assert 0 <= routed <= top
