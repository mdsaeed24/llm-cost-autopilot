"""scripts/train_classifier.py end to end on synthetic labels."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from autopilot.classifier.registry import list_versions, read_registry
from tests.conftest import load_script
from tests.factories import write_synthetic_labels


def run(tmp_path: Path, *extra: str) -> str:
    labels = tmp_path / "labels.jsonl"
    if not labels.exists():
        write_synthetic_labels(labels)
    buffer = io.StringIO()
    code = load_script("train_classifier").main(
        ["--labels", str(labels), "--split", str(tmp_path / "split.json"),
         "--models-dir", str(tmp_path / "models"), "--folds", "3", *extra],
        console=Console(file=buffer, width=200),
    )
    assert code == 0
    return buffer.getvalue()


def test_first_run_saves_and_promotes_v1(tmp_path: Path) -> None:
    out = run(tmp_path)
    assert "created and frozen" in out
    assert "Cross-validation" in out and "Confusion matrix" in out and "Route-up threshold" in out
    assert read_registry(tmp_path / "models").current == "v1"


def test_later_runs_reuse_split_and_compare_with_current(tmp_path: Path) -> None:
    run(tmp_path)
    split_before = (tmp_path / "split.json").read_text()

    out = run(tmp_path, "--no-promote")
    assert "reused (frozen)" in out
    assert (tmp_path / "split.json").read_text() == split_before
    assert "Current model v1 on this test set" in out
    assert list_versions(tmp_path / "models") == [1, 2]
    assert read_registry(tmp_path / "models").current == "v1"

    out = run(tmp_path)  # same data → same scores → "not worse" → promoted by the rule
    assert read_registry(tmp_path / "models").current == "v3"
    assert "not worse" in read_registry(tmp_path / "models").history[-1].reason


def test_tfidf_only_flag_skips_embedding_candidates(tmp_path: Path) -> None:
    out = run(tmp_path, "--feature-sets", "tfidf_hc")
    assert "embed_hc" not in out
    assert "logistic_regression+tfidf_hc" in out
