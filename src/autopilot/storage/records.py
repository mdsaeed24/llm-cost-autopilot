"""Typed rows written to the database."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from autopilot.llm.schemas import ChatMessage

VerificationStatus = Literal[
    "not_sampled",         # eligible, but not picked by the random sample
    "skipped_top_tier",    # already answered by Tier 3: nothing to compare against
    "excluded_truncated",  # routed answer hit max_tokens: excluded from quality parity
    "queued",              # waiting for the background worker
    "verified",            # judged (pass or fail)
    "excluded",            # verification ran but could not be judged (e.g. reference truncated)
    "failed",              # verification job errored out after all retries
    "error",               # the request itself failed
]
VerificationSource = Literal["sampled", "sync"]
EscalationMode = Literal["sync", "async"]


class RequestRecord(BaseModel):
    """One row of the ``requests`` table."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    created_at: datetime
    prompt_hash: str
    mode: Literal["normal", "evaluation"]
    predicted_tier: int | None
    confidence: float | None
    routed_tier: int
    routed_up: bool
    classifier_version: str | None
    task_type: str | None
    task_confidence: float | None
    routing_reason: str
    routed_model: str
    served_tier: int | None
    served_model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    serve_cost_usd: float = 0.0
    verification_cost_usd: float = 0.0
    latency_ms: float | None = None
    finish_reason: str | None = None
    truncated: bool = False
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    verification_status: VerificationStatus
    verified: bool = False
    judge_score: int | None = None
    quality_passed: bool | None = None
    escalated: bool = False


class JobPayload(BaseModel):
    """Everything the worker needs to verify one request, without re-reading other tables."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    prompt: str = Field(description="The text the router classified (last user message).")
    routed_output: str
    served_model: str
    served_tier: int
    served_cost_usd: float
    task_type: str | None
    task_confidence: float | None


class Job(BaseModel):
    """A claimed queue entry."""

    model_config = ConfigDict(extra="forbid")

    id: int
    request_id: str
    attempts: int
    payload: JobPayload


class VerificationRecord(BaseModel):
    """One row of the ``verifications`` table."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: datetime
    source: VerificationSource
    reference_model: str
    reference_output: str | None
    reference_cost_usd: float
    reference_latency_ms: float | None
    reference_truncated: bool
    judge_model: str | None = None
    judge_cost_usd: float = 0.0
    routed_score: int | None = None
    reference_score: int | None = None
    agreement: int | None = None
    judge_reason: str | None = None
    rule: str | None = None
    pass_threshold: int | None = None
    passed: bool | None = None
    excluded_reason: str | None = None

    @property
    def total_cost_usd(self) -> float:
        return self.reference_cost_usd + self.judge_cost_usd


class EscalationRecord(BaseModel):
    """One row of the ``escalations`` table."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: datetime
    mode: EscalationMode
    original_model: str
    escalated_model: str
    original_cost_usd: float
    escalated_cost_usd: float
    cost_delta_usd: float
    judge_score: int | None
    quality_gap: int | None
    answer_replaced: bool


class RoutingFailureRecord(BaseModel):
    """One row of the ``routing_failures`` table."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: datetime
    prompt: str
    served_tier: int
    correct_tier: int
    task_type: str | None
    judge_score: int | None
    judge_reason: str | None
    used_in_version: str | None = None
    id: int | None = None
