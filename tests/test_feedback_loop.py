"""Routing failures → training examples → retrain → compare → promote (or not) → log."""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

from autopilot.classifier.feedback import failures_to_labels
from autopilot.classifier.registry import read_registry
from autopilot.storage.db import Database, utc_now
from autopilot.storage.records import RoutingFailureRecord
from autopilot.storage.repository import Repository
from tests.conftest import load_script
from tests.factories import write_synthetic_labels
from tests.test_storage import request


def failure(rid: str, prompt: str, served_tier: int = 1, task_type: str | None = "reasoning_creative") -> RoutingFailureRecord:
    return RoutingFailureRecord(request_id=rid, created_at=utc_now(), prompt=prompt, served_tier=served_tier,
                                correct_tier=min(3, served_tier + 1), task_type=task_type, judge_score=2,
                                judge_reason="weak", id=hash(rid) % 10_000)


def test_failures_to_labels_guards() -> None:
    failures = [
        failure("a", "Why do cats purr? Explain."),
        failure("b", "  why do CATS purr? explain. "),     # duplicate after normalising
        failure("c", "In the test set"),
        failure("d", "Labeled by a human"),
        failure("e", "No task type", task_type=None),
        failure("f", "Summarise the meeting", served_tier=2),
    ]
    result = failures_to_labels(failures, test_prompts=["in the test set"], human_prompts=["Labeled by a human"])
    assert [(e.prompt, e.tier, e.source) for e in result.examples] == [
        ("Why do cats purr? Explain.", 2, "routing_failure"),
        ("Summarise the meeting", 3, "routing_failure"),
    ]
    assert (result.dropped_duplicate, result.dropped_test_overlap, result.dropped_human_overlap,
            result.dropped_no_task_type) == (1, 1, 1, 1)
    assert result.examples[0].candidate_id.startswith("fail-")


def run(script: str, *args: str) -> str:
    buffer = io.StringIO()
    assert load_script(script).main(list(args), console=Console(file=buffer, width=200)) == 0
    return buffer.getvalue()


def test_retrain_end_to_end(tmp_path: Path) -> None:
    labels = write_synthetic_labels(tmp_path / "labels.jsonl")
    split, models = tmp_path / "split.json", tmp_path / "models"
    common = ["--labels", str(labels), "--split", str(split), "--models-dir", str(models), "--folds", "3",
              "--feature-sets", "tfidf_hc"]
    run("train_classifier", *common)  # v1 + frozen split
    assert read_registry(models).current == "v1"

    db = tmp_path / "flow.db"
    repo = Repository(Database(db))
    for i in range(3):
        rid = f"req{i}"
        repo.insert_request(request(rid, verification_status="verified"))
        with repo.db.connect() as conn:
            conn.execute(
                "INSERT INTO routing_failures (request_id, created_at, prompt, served_tier, correct_tier, task_type,"
                " judge_score, judge_reason) VALUES (?, ?, ?, 1, 2, 'reasoning_creative', 2, 'weak')",
                (rid, "2026-10-16T00:00:00.000000Z", f"Brainstorm {i} creative uses for a paperclip and explain why"),
            )

    out = run("retrain", *common, "--db", str(db), "--failures-out", str(tmp_path / "failures.jsonl"))
    assert "usable examples: 3 (3 new)" in out
    log = [json.loads(line) for line in (models / "retrain_log.jsonl").read_text().splitlines()]
    assert log[-1]["candidate_version"] == "v2" and log[-1]["failure_examples"] == 3
    assert log[-1]["current"] is not None  # compared against v1 on the same test set
    assert all(f.used_in_version == "v2" for f in repo.list_routing_failures())
    assert len((tmp_path / "failures.jsonl").read_text().splitlines()) == 3

    out = run("retrain", *common, "--db", str(db), "--failures-out", str(tmp_path / "failures.jsonl"))
    assert "nothing to retrain" in out  # no new failures since v2

    out = run("retrain", *common, "--db", str(db), "--failures-out", str(tmp_path / "failures.jsonl"),
              "--min-new-failures", "0", "--dry-run")
    assert "Dry run" in out
    assert not (models / "classifier_v3.joblib").exists()
