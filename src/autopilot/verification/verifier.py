"""Verify one routed answer against Tier 3, and turn the outcome into database records."""

from __future__ import annotations

import random
from dataclasses import dataclass

from autopilot.llm.catalog import ModelCatalog
from autopilot.llm.client import LLMClient
from autopilot.llm.schemas import LLMResponse
from autopilot.routing.config import RoutingConfig
from autopilot.settings import VerificationSettings
from autopilot.storage.db import utc_now
from autopilot.storage.records import (
    EscalationMode,
    EscalationRecord,
    JobPayload,
    RoutingFailureRecord,
    VerificationRecord,
    VerificationSource,
)
from autopilot.verification.judge import JudgeError, judge
from autopilot.verification.rules import QualityConfig

TOP_TIER = 3


@dataclass
class VerificationOutcome:
    """The verification row plus the reference answer (sync mode may return it to the user)."""

    record: VerificationRecord
    reference: LLMResponse

    @property
    def failed(self) -> bool:
        return self.record.passed is False


class Verifier:
    """Gets the Tier-3 answer for the same messages and asks the judge to compare."""

    def __init__(
        self,
        client: LLMClient,
        catalog: ModelCatalog,
        routing: RoutingConfig,
        quality: QualityConfig,
        settings: VerificationSettings,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.client = client
        self.catalog = catalog
        self.routing = routing
        self.quality = quality
        self.settings = settings
        self.rng = rng or random.Random()

    async def verify(self, request_id: str, payload: JobPayload, *, source: VerificationSource) -> VerificationOutcome:
        """Raises LLMRequestError if the reference or judge call fails (the worker retries)."""
        reference_model = self.catalog.get(self.routing.tiers[TOP_TIER])
        reference = await self.client.send(payload.messages, reference_model)
        rule_name, rule = self.quality.select(payload.task_type, payload.task_confidence)
        base = dict(
            request_id=request_id,
            created_at=utc_now(),
            source=source,
            reference_model=reference_model.name,
            reference_output=reference.output_text,
            reference_cost_usd=reference.cost_usd,
            reference_latency_ms=reference.latency_ms,
            reference_truncated=reference.truncated,
            rule=rule_name,
            pass_threshold=rule.pass_threshold,
        )
        if reference.truncated:
            # A cut-off reference is not a fair yardstick: excluded from parity, cost still counted.
            return VerificationOutcome(VerificationRecord(**base, excluded_reason="reference_truncated"), reference)

        judge_model = self.catalog.get(self.quality.judge_model)
        try:
            result = await judge(
                self.client,
                judge_model,
                messages=payload.messages,
                routed_answer=payload.routed_output,
                reference_answer=reference.output_text,
                rule=rule,
                rng=self.rng,
                max_tokens=self.settings.judge_max_tokens,
            )
        except JudgeError as error:
            record = VerificationRecord(
                **base, judge_model=judge_model.name, judge_cost_usd=error.response.cost_usd,
                excluded_reason="judge_unparseable",
            )
            return VerificationOutcome(record, reference)

        record = VerificationRecord(
            **base,
            judge_model=judge_model.name,
            judge_cost_usd=result.response.cost_usd,
            routed_score=result.routed_score,
            reference_score=result.reference_score,
            agreement=result.agreement,
            judge_reason=result.reason,
            passed=result.agreement >= rule.pass_threshold,
        )
        return VerificationOutcome(record, reference)


def consequences(
    request_id: str, payload: JobPayload, outcome: VerificationOutcome, *, mode: EscalationMode
) -> tuple[EscalationRecord | None, RoutingFailureRecord | None]:
    """A failed verification becomes an escalation entry and a routing-failure training example.

    ``mode="sync"`` means the user received the Tier-3 answer instead; ``"async"`` means the
    failure is only logged (the user already has the routed answer).
    """
    record = outcome.record
    if not outcome.failed:
        return None, None
    assert record.agreement is not None and record.pass_threshold is not None
    now = utc_now()
    escalation = EscalationRecord(
        request_id=request_id,
        created_at=now,
        mode=mode,
        original_model=payload.served_model,
        escalated_model=record.reference_model,
        original_cost_usd=payload.served_cost_usd,
        escalated_cost_usd=record.reference_cost_usd,
        cost_delta_usd=record.reference_cost_usd - payload.served_cost_usd,
        judge_score=record.agreement,
        quality_gap=record.pass_threshold - record.agreement,
        answer_replaced=mode == "sync",
    )
    failure = RoutingFailureRecord(
        request_id=request_id,
        created_at=now,
        prompt=payload.prompt,
        served_tier=payload.served_tier,
        correct_tier=min(TOP_TIER, payload.served_tier + 1),
        task_type=payload.task_type,
        judge_score=record.agreement,
        judge_reason=record.judge_reason,
    )
    return escalation, failure
