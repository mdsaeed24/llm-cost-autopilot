"""The request flow: route → answer → (verify) → log. Used by the CLI now and the API in Phase 5."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
import uuid
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from autopilot.llm.catalog import ModelCatalog
from autopilot.llm.client import LLMClient, LLMRequestError, MessageLike
from autopilot.llm.schemas import ChatMessage, LLMResponse, MissingAPIKeyError
from autopilot.routing.router import Router, RoutingDecision
from autopilot.settings import RunMode, VerificationSettings
from autopilot.storage.db import utc_now
from autopilot.storage.records import JobPayload, RequestRecord, VerificationRecord, VerificationStatus
from autopilot.storage.repository import Repository
from autopilot.verification.sampling import plan_verification
from autopilot.verification.verifier import VerificationOutcome, Verifier, consequences

logger = logging.getLogger(__name__)

TOP_TIER = 3


class CompletionResult(BaseModel):
    """What a caller gets back: the answer plus how it was produced."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    request_id: str
    output_text: str
    decision: RoutingDecision
    served_model: str
    served_tier: int
    response: LLMResponse
    failover: bool
    escalated: bool
    verification_status: VerificationStatus
    verification: VerificationRecord | None
    latency_ms: float


def routing_text(messages: Sequence[ChatMessage]) -> str:
    """The text the classifier sees: the latest user message."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise ValueError("messages must contain at least one user message")


def prompt_hash(text: str) -> str:
    """Logged instead of the prompt itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AutopilotService:
    """Wires router, LLM client, verifier and repository together."""

    def __init__(
        self,
        *,
        router: Router,
        client: LLMClient,
        repo: Repository,
        verifier: Verifier,
        settings: VerificationSettings,
        rng: random.Random | None = None,
    ) -> None:
        self.router = router
        self.client = client
        self.repo = repo
        self.verifier = verifier
        self.settings = settings
        self.rng = rng or random.Random()

    @property
    def catalog(self) -> ModelCatalog:
        return self.router.catalog

    @property
    def mode(self) -> RunMode:
        return self.settings.mode

    async def complete(
        self,
        messages: Sequence[MessageLike],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Answer ``messages`` with the cheapest suitable model and log everything."""
        started = time.perf_counter()
        chat = [m if isinstance(m, ChatMessage) else ChatMessage.model_validate(m) for m in messages]
        prompt = routing_text(chat)
        # Classification may call a local model (embeddings) — keep it off the event loop.
        decision = await asyncio.to_thread(self.router.decide, prompt)
        request_id = uuid.uuid4().hex

        try:
            response, served_tier = await self._answer(chat, decision.routed_tier, max_tokens, temperature)
        except (LLMRequestError, MissingAPIKeyError) as error:
            self.repo.insert_request(self._record(request_id, prompt, decision, None, None, started,
                                                  status="error", error=str(error), verification_status="error"))
            raise
        served_model = self.router.config.tiers[served_tier]  # type: ignore[index]
        payload = JobPayload(
            messages=chat,
            prompt=prompt,
            routed_output=response.output_text,
            served_model=served_model,
            served_tier=served_tier,
            served_cost_usd=response.cost_usd,
            task_type=decision.task_type,
            task_confidence=decision.task_confidence,
        )

        outcome = await self._maybe_verify_sync(request_id, payload, decision, response, served_tier)
        escalated = outcome is not None and outcome.failed
        final = outcome.reference if (outcome is not None and escalated) else response

        if outcome is not None:
            status: VerificationStatus = "verified" if outcome.record.passed is not None else "excluded"
        else:
            status = plan_verification(
                served_tier=served_tier, truncated=response.truncated,
                sample_rate=self.settings.sample_rate, rng=self.rng,
            )
        record = self._record(request_id, prompt, decision, response, served_tier, started,
                              verification_status=status, served_model=served_model)
        self.repo.insert_request(record, job=payload if status == "queued" else None)
        if outcome is not None:
            escalation, failure = consequences(request_id, payload, outcome, mode="sync")
            self.repo.save_verification(outcome.record, escalation=escalation, failure=failure)

        return CompletionResult(
            request_id=request_id,
            output_text=final.output_text,
            decision=decision,
            served_model=final.model_name,
            served_tier=TOP_TIER if escalated else served_tier,
            response=final,
            failover=served_tier != decision.routed_tier,
            escalated=escalated,
            verification_status=status,
            verification=outcome.record if outcome else None,
            latency_ms=record.latency_ms or 0.0,
        )

    async def _answer(
        self, chat: list[ChatMessage], tier: int, max_tokens: int | None, temperature: float | None
    ) -> tuple[LLMResponse, int]:
        """Call the tier's model; if it is down, fail over to the next tier up."""
        while True:
            model = self.catalog.get(self.router.config.tiers[tier])  # type: ignore[index]
            try:
                response = await self.client.send(chat, model, max_tokens=max_tokens, temperature=temperature)
                return response, tier
            except LLMRequestError as error:
                if tier >= TOP_TIER:
                    raise
                logger.warning("Tier %d (%s) failed: %s — failing over to tier %d", tier, model.name, error, tier + 1)
                tier += 1

    async def _maybe_verify_sync(
        self,
        request_id: str,
        payload: JobPayload,
        decision: RoutingDecision,
        response: LLMResponse,
        served_tier: int,
    ) -> VerificationOutcome | None:
        """Sync escalation: for low-confidence requests, check against Tier 3 before responding."""
        low_confidence = decision.confidence is None or decision.confidence < self.settings.sync_confidence_below
        if not (self.settings.escalation_mode == "sync" and low_confidence
                and served_tier < TOP_TIER and not response.truncated):
            return None
        try:
            return await self.verifier.verify(request_id, payload, source="sync")
        except LLMRequestError as error:
            # Tier 3 is unavailable: keep the routed answer rather than failing the request.
            logger.warning("Sync verification failed for %s: %s — returning routed answer", request_id, error)
            return None

    def _record(
        self,
        request_id: str,
        prompt: str,
        decision: RoutingDecision,
        response: LLMResponse | None,
        served_tier: int | None,
        started: float,
        *,
        verification_status: VerificationStatus,
        served_model: str | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> RequestRecord:
        return RequestRecord(
            id=request_id,
            created_at=utc_now(),
            prompt_hash=prompt_hash(prompt),
            mode=self.settings.mode,
            predicted_tier=decision.predicted_tier,
            confidence=decision.confidence,
            routed_tier=decision.routed_tier,
            routed_up=decision.routed_up,
            classifier_version=decision.classifier_version,
            task_type=decision.task_type,
            task_confidence=decision.task_confidence,
            routing_reason=decision.reason,
            routed_model=decision.model_name,
            served_tier=served_tier,
            served_model=served_model,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            serve_cost_usd=response.cost_usd if response else 0.0,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            finish_reason=response.finish_reason if response else None,
            truncated=response.truncated if response else False,
            status=status,  # type: ignore[arg-type]
            error=error,
            verification_status=verification_status,
        )
