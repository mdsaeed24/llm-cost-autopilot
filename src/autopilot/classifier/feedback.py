"""Turn confirmed routing failures into labeled training examples.

A routing failure means the tier that answered was too weak, so the example's label is
"one tier up". Two guards keep the data honest:
- prompts that appear in the frozen test set are dropped (otherwise we'd train on the test);
- prompts that a human already labeled are dropped (the human label wins).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from autopilot.dataset.schemas import TASK_TYPES, LabeledPrompt
from autopilot.storage.records import RoutingFailureRecord


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class FeedbackExamples:
    examples: list[LabeledPrompt]
    failure_ids: list[int]
    dropped_test_overlap: int
    dropped_human_overlap: int
    dropped_duplicate: int
    dropped_no_task_type: int


def failures_to_labels(
    failures: Sequence[RoutingFailureRecord],
    *,
    test_prompts: Iterable[str],
    human_prompts: Iterable[str],
) -> FeedbackExamples:
    """Build ``source="routing_failure"`` labels from failure records."""
    test = {normalise(p) for p in test_prompts}
    human = {normalise(p) for p in human_prompts}
    seen: set[str] = set()
    result = FeedbackExamples([], [], 0, 0, 0, 0)
    for failure in failures:
        key = normalise(failure.prompt)
        if key in test:
            result.dropped_test_overlap += 1
            continue
        if key in human:
            result.dropped_human_overlap += 1
            continue
        if key in seen:
            result.dropped_duplicate += 1
            continue
        if failure.task_type not in TASK_TYPES:
            result.dropped_no_task_type += 1
            continue
        seen.add(key)
        result.examples.append(
            LabeledPrompt(
                candidate_id=f"fail-{failure.request_id[:12]}",
                prompt=failure.prompt,
                tier=failure.correct_tier,  # type: ignore[arg-type]
                task_type=failure.task_type,  # type: ignore[arg-type]
                note=f"routing failure: tier {failure.served_tier} judged {failure.judge_score}/5 — {failure.judge_reason}",
                source="routing_failure",
                labeled_at=failure.created_at,
            )
        )
        if failure.id is not None:
            result.failure_ids.append(failure.id)
    return result
