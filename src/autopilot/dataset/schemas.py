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


# Decision rules for labeling round 2. Go top to bottom; the first rule that matches decides.
# Round 1 used only TIER_HELP, which left common cases (e.g. a fact question with no text
# given) undefined — similar prompts ended up in different tiers.
RUBRIC_VERSION = "rules-v2"
RUBRIC: tuple[tuple[str, Tier], ...] = (
    ("The answer is already in the given text (find, copy, reformat, fix typos)", 1),
    ("A short fact question answerable in one line (who / what / when / where)", 1),
    ("Summarise, classify or sort things, make a list, or give a structured overview", 2),
    ("Asks why/how, wants advice or an opinion weighing pros and cons, needs multi-step "
     "reasoning or maths, or asks for original creative writing", 3),
)
RUBRIC_TIE_BREAK = "Torn between two tiers? Pick the LOWER one — the router already routes unsure prompts up."


def rubric_text() -> str:
    """The rubric as numbered lines, for display."""
    lines = [f"{i}. {rule}  → Tier {tier}" for i, (rule, tier) in enumerate(RUBRIC, start=1)]
    return "\n".join([*lines, f"{len(RUBRIC) + 1}. {RUBRIC_TIE_BREAK}"])


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
