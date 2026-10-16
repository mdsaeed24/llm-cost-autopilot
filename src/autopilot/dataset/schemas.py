"""Records for prompt candidates, human labels and the load-test pool."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal[1, 2, 3]
TaskType = Literal[
    "extraction_reformatting",
    "qa_classification",
    "summarization_analysis",
    "reasoning_creative",
]
LabelSource = Literal["human", "routing_failure"]

TIERS: tuple[Tier, ...] = get_args(Tier)
TASK_TYPES: tuple[TaskType, ...] = get_args(TaskType)

TIER_HELP: dict[Tier, str] = {
    1: "simple — reformatting, extraction, basic Q&A answerable from provided context",
    2: "moderate — summarization, classification, structured analysis",
    3: "complex — multi-step reasoning, creative generation, nuanced judgment",
}

TASK_TYPE_HELP: dict[TaskType, str] = {
    "extraction_reformatting": "pull facts out of given text, or change its format",
    "qa_classification": "answer a question, or assign a label/category",
    "summarization_analysis": "condense text, or break it down / compare in a structured way",
    "reasoning_creative": "multi-step reasoning, open-ended judgment, or creative writing",
}


# Decision rules for labeling. Go top to bottom; the first rule that matches decides.
# rules-v2 (round 2) fixed round 1's undefined cases. rules-v3 (round 3) changes ONE rule after
# the load test showed the 3B local model answers memory-only fact questions correctly only
# ~61% of the time (vs ~97% for Flash): such questions now go to Tier 2.
RubricRules = tuple[tuple[str, Tier], ...]

_RULE_IN_TEXT = ("The answer is already in the given text (find, copy, reformat, fix typos)", 1)
_RULE_STRUCTURE = ("Summarise, classify or sort things, make a list, or give a structured overview", 2)
_RULE_HARD = (
    "Asks why/how, wants advice or an opinion weighing pros and cons, needs multi-step "
    "reasoning or maths, or asks for original creative writing",
    3,
)

RUBRICS: dict[str, RubricRules] = {
    "rules-v2": (
        _RULE_IN_TEXT,
        ("A short fact question answerable in one line (who / what / when / where)", 1),
        _RULE_STRUCTURE,
        _RULE_HARD,
    ),
    "rules-v3": (
        _RULE_IN_TEXT,
        ("A fact question answered from memory — no text given to find it in (who / what / when / where)", 2),
        _RULE_STRUCTURE,
        _RULE_HARD,
    ),
}
RUBRIC_VERSION = "rules-v2"   # used by labeling round 2
LATEST_RUBRIC = "rules-v3"    # used by round 3 (memory-fact review)
RUBRIC_TIE_BREAK = "Torn between two tiers? Pick the LOWER one — the router already routes unsure prompts up."


def rubric_text(version: str = RUBRIC_VERSION) -> str:
    """The rubric as numbered lines, for display."""
    rules = RUBRICS[version]
    lines = [f"{i}. {rule}  → Tier {tier}" for i, (rule, tier) in enumerate(rules, start=1)]
    return "\n".join([*lines, f"{len(rules) + 1}. {RUBRIC_TIE_BREAK}"])


def _now() -> datetime:
    return datetime.now(UTC)


class Candidate(BaseModel):
    """An unlabeled prompt shown to the human labeler. Deliberately has no category or hint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class LoadtestPrompt(BaseModel):
    """An unlabeled prompt reserved for the Phase 6 load test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loadtest_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class LabeledPrompt(BaseModel):
    """A prompt with a tier and task type assigned by a human (or, later, by the feedback loop)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    tier: Tier
    task_type: TaskType
    note: str | None = None
    source: LabelSource = "human"
    rubric: str | None = Field(default=None, description="Rubric version used; None = round 1 (no rubric).")
    labeled_at: datetime = Field(default_factory=_now)


class SkippedCandidate(BaseModel):
    """A candidate the labeler rejected as a bad prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    note: str | None = None
    skipped_at: datetime = Field(default_factory=_now)
