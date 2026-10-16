"""Labeling round 2: re-label with the written rubric, then measure self-agreement.

Self-agreement (how often round-2 labels match round-1 labels for the same prompt) is a
practical ceiling for any classifier: a model can't be more consistent than its labels.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence

from pydantic import BaseModel, ConfigDict

from autopilot.dataset.schemas import TIERS, Candidate, LabeledPrompt


def round2_queue(
    candidates: Sequence[Candidate],
    previous: Sequence[LabeledPrompt],
    test_ids: Collection[str],
    *,
    seed: int = 7,
) -> list[Candidate]:
    """Order for round 2: frozen test prompts, then other round-1 prompts, then never-labeled ones.

    Each group is shuffled, and nothing on screen says which group a prompt is in. Test prompts
    come first so that stopping early still leaves a complete, comparable test set.
    """
    rng = random.Random(seed)
    previously_labeled = {label.candidate_id for label in previous}
    groups: list[list[Candidate]] = [
        [c for c in candidates if c.candidate_id in test_ids],
        [c for c in candidates if c.candidate_id in previously_labeled and c.candidate_id not in test_ids],
        [c for c in candidates if c.candidate_id not in previously_labeled and c.candidate_id not in test_ids],
    ]
    queue: list[Candidate] = []
    for group in groups:
        rng.shuffle(group)
        queue.extend(group)
    return queue


class Agreement(BaseModel):
    """How round-2 labels compare with round-1 labels on the prompts labeled in both rounds."""

    model_config = ConfigDict(extra="forbid")

    n: int
    same: int
    rate: float
    moved_up: int    # round 2 tier is higher than round 1
    moved_down: int
    confusion: list[list[int]]  # rows = round-1 tier, columns = round-2 tier


def agreement(previous: Sequence[LabeledPrompt], current: Sequence[LabeledPrompt]) -> Agreement:
    """Compare the two rounds on their overlapping candidate ids."""
    before = {label.candidate_id: label.tier for label in previous}
    pairs = [(before[label.candidate_id], label.tier) for label in current if label.candidate_id in before]
    confusion = [[sum(1 for a, b in pairs if a == r and b == c) for c in TIERS] for r in TIERS]
    same = sum(a == b for a, b in pairs)
    return Agreement(
        n=len(pairs),
        same=same,
        rate=same / len(pairs) if pairs else 0.0,
        moved_up=sum(b > a for a, b in pairs),
        moved_down=sum(b < a for a, b in pairs),
        confusion=confusion,
    )
