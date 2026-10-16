"""Label store: save-after-every-answer, resume, undo, validation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from autopilot.dataset.io import DataFileError
from autopilot.dataset.labels import LabelStore
from autopilot.dataset.schemas import LabeledPrompt, SkippedCandidate

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def label(cid: str, tier: int = 1, *, at: datetime = T0) -> LabeledPrompt:
    return LabeledPrompt(candidate_id=cid, prompt=f"prompt {cid}", tier=tier, task_type="qa_classification", labeled_at=at)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> LabelStore:
    return LabelStore(tmp_path / "labeled.jsonl", tmp_path / "skipped.jsonl")


def test_every_change_is_persisted_and_resumable(store: LabelStore, tmp_path: Path) -> None:
    store.add_label(label("c1", 3))
    store.add_skip(SkippedCandidate(candidate_id="c2", note="broken"))

    reopened = LabelStore(tmp_path / "labeled.jsonl", tmp_path / "skipped.jsonl")
    assert [r.candidate_id for r in reopened.labels] == ["c1"]
    assert reopened.labels[0].tier == 3
    assert reopened.labels[0].source == "human"
    assert reopened.done_ids == {"c1", "c2"}
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes leave no temp files


def test_labels_file_contains_only_labels(store: LabelStore, tmp_path: Path) -> None:
    store.add_label(label("c1"))
    store.add_skip(SkippedCandidate(candidate_id="c2"))
    assert (tmp_path / "labeled.jsonl").read_text().count("\n") == 1
    assert "c2" not in (tmp_path / "labeled.jsonl").read_text()


def test_duplicate_labels_are_rejected(store: LabelStore) -> None:
    store.add_label(label("c1"))
    with pytest.raises(ValueError, match="already"):
        store.add_skip(SkippedCandidate(candidate_id="c1"))


def test_undo_removes_most_recent_action(store: LabelStore, tmp_path: Path) -> None:
    store.add_label(label("c1", at=T0))
    store.add_skip(SkippedCandidate(candidate_id="c2", skipped_at=T0 + timedelta(seconds=1)))
    store.add_label(label("c3", at=T0 + timedelta(seconds=2)))

    assert store.undo_last() == "c3"
    assert store.undo_last() == "c2"
    assert store.undo_last() == "c1"
    assert store.undo_last() is None
    assert LabelStore(tmp_path / "labeled.jsonl", tmp_path / "skipped.jsonl").done_ids == set()


def test_counts(store: LabelStore) -> None:
    for i, tier in enumerate([1, 3, 3, 2]):
        store.add_label(label(f"c{i}", tier))
    assert store.tier_counts() == {1: 1, 2: 1, 3: 2}
    assert store.task_type_counts() == {"qa_classification": 4}


def test_corrupt_file_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "labeled.jsonl"
    path.write_text(label("c1").model_dump_json() + "\n{not json}\n")
    with pytest.raises(DataFileError, match=":2:"):
        LabelStore(path, tmp_path / "skipped.jsonl")


@pytest.mark.parametrize("bad", [{"tier": 4}, {"task_type": "reasoning"}, {"tier": 0}])
def test_invalid_labels_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LabeledPrompt.model_validate({"candidate_id": "c1", "prompt": "p", "tier": 1, "task_type": "qa_classification", **bad})
