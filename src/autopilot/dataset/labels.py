"""Persistent store of human labels and skips, saved after every change."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from autopilot.dataset.io import read_jsonl, write_jsonl
from autopilot.dataset.schemas import LabeledPrompt, SkippedCandidate


class LabelStore:
    """Labels live in one JSONL file, skips in another. Every mutation is written immediately.

    Keeping skips out of the labels file means training code can read
    ``labeled_prompts.jsonl`` as-is without filtering.
    """

    def __init__(self, labels_path: Path, skips_path: Path) -> None:
        self.labels_path = labels_path
        self.skips_path = skips_path
        self.labels: list[LabeledPrompt] = read_jsonl(labels_path, LabeledPrompt)
        self.skips: list[SkippedCandidate] = read_jsonl(skips_path, SkippedCandidate)

    @property
    def done_ids(self) -> set[str]:
        """Candidates already labeled or skipped (these are not shown again)."""
        return {r.candidate_id for r in self.labels} | {s.candidate_id for s in self.skips}

    def add_label(self, label: LabeledPrompt) -> None:
        self._ensure_new(label.candidate_id)
        self.labels.append(label)
        write_jsonl(self.labels_path, self.labels)

    def add_skip(self, skip: SkippedCandidate) -> None:
        self._ensure_new(skip.candidate_id)
        self.skips.append(skip)
        write_jsonl(self.skips_path, self.skips)

    def undo_last(self) -> str | None:
        """Remove the most recent label or skip. Returns its candidate id, or None if empty."""
        last_label = self.labels[-1] if self.labels else None
        last_skip = self.skips[-1] if self.skips else None
        if last_label is None and last_skip is None:
            return None
        if last_skip is None or (last_label is not None and last_label.labeled_at >= last_skip.skipped_at):
            assert last_label is not None
            self.labels.pop()
            write_jsonl(self.labels_path, self.labels)
            return last_label.candidate_id
        self.skips.pop()
        write_jsonl(self.skips_path, self.skips)
        return last_skip.candidate_id

    def tier_counts(self) -> Counter[int]:
        return Counter(r.tier for r in self.labels)

    def task_type_counts(self) -> Counter[str]:
        return Counter(r.task_type for r in self.labels)

    def _ensure_new(self, candidate_id: str) -> None:
        if candidate_id in self.done_ids:
            raise ValueError(f"{candidate_id} is already labeled or skipped")
