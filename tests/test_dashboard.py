"""The Streamlit app runs headlessly against real (temporary) databases and never shows savings alone."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from autopilot.settings import PROJECT_ROOT
from autopilot.storage.db import Database
from autopilot.storage.repository import Repository
from tests.test_metrics_honest import db as populated_db  # noqa: F401 - fixture reuse
from tests.test_storage import request

APP = str(PROJECT_ROOT / "src" / "autopilot" / "dashboard" / "app.py")
SAVINGS = ("Net savings (after verification)", "Gross savings (before verification)")


def run_app(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> AppTest:
    monkeypatch.setenv("AUTOPILOT_DB_PATH", str(db_path))
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_dashboard_with_data(monkeypatch: pytest.MonkeyPatch, populated_db: Database) -> None:  # noqa: F811
    app = run_app(monkeypatch, populated_db.path)
    metrics = {m.label: m for m in app.metric}
    # Savings are always accompanied by quality parity and verification overhead.
    for label in (*SAVINGS, "Quality parity", "Verification overhead"):
        assert label in metrics
    assert metrics["Quality parity"].value == "50%"
    assert "n=2" in metrics["Quality parity"].delta
    assert "quality parity" in app.markdown[0].value
    assert any("Small sample" in w.value for w in app.warning)
    caption = next(c.value for c in app.caption if "Excluded from parity" in c.value)
    assert "\\$" in caption and "$" not in caption.replace("\\$", "")  # no bare $ → no accidental LaTeX
    # Money changes carry their sign first, so a loss gets a red down arrow (not "$-1.21" shown as a gain).
    for label in SAVINGS:
        assert metrics[label].delta[0] in "+-" and metrics[label].delta[1] == "$"
    assert [h.value for h in app.subheader] == [
        "Cost per day", "Actual vs everything on Tier 3", "Routing distribution", "Quality score distribution",
        "Escalation rate over time", "Latency by tier", "Truncation by tier",
    ]


def test_savings_hidden_until_quality_is_measured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database = Database(tmp_path / "unverified.db")
    Repository(database).insert_request(request("x", input_tokens=10, output_tokens=10, latency_ms=100,
                                                verification_status="not_sampled"))
    app = run_app(monkeypatch, database.path)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Quality parity"] == "Not measured"
    assert all(metrics[label] == "—" for label in SAVINGS)


def test_empty_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    Database(tmp_path / "empty.db")
    app = run_app(monkeypatch, tmp_path / "empty.db")
    assert any("No requests" in i.value for i in app.info)
    assert not app.metric
