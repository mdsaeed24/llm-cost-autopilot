"""End-to-end Phase 3 flow with fake LLMs: service → log → queue → worker → judge → feedback."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from autopilot.classifier.predict import Prediction
from autopilot.llm.catalog import ModelCatalog, load_catalog
from autopilot.llm.client import LLMRequestError
from autopilot.routing.config import ConfidenceRule, RoutingConfig
from autopilot.routing.router import Router
from autopilot.service import AutopilotService, routing_text
from autopilot.settings import VerificationSettings
from autopilot.storage.db import Database
from autopilot.storage.repository import Repository
from autopilot.verification.rules import load_quality_config
from autopilot.verification.verifier import Verifier
from autopilot.verification.worker import Worker
from tests.conftest import FakeOpenAI, connection_error, make_client, make_completion

ANSWERS = {"llama3.2:3b": "CHEAP-ANSWER", "deepseek-flash": "MID-ANSWER", "deepseek-v4-pro": "REFERENCE-ANSWER"}


class World:
    """Configurable fake LLM backend."""

    def __init__(self) -> None:
        self.routed_score = 5
        self.judge_reply: str | None = None
        self.truncate: set[str] = set()
        self.down: set[str] = set()

    async def __call__(self, kwargs: dict[str, Any]) -> Any:
        model = kwargs["model"]
        if model in self.down:
            raise connection_error()
        system = kwargs["messages"][0]
        if system["role"] == "system" and "impartial evaluator" in system["content"]:
            if self.judge_reply is not None:
                return make_completion(self.judge_reply)
            user = kwargs["messages"][1]["content"]
            reference_is_a = "REFERENCE-ANSWER" in user.split("<answer_a>")[1].split("</answer_a>")[0]
            a, b = (5, self.routed_score) if reference_is_a else (self.routed_score, 5)
            return make_completion(json.dumps({"score_a": a, "score_b": b, "reason": "judged"}))
        finish = "length" if model in self.truncate else "stop"
        return make_completion(ANSWERS[model], finish_reason=finish, prompt_tokens=200, completion_tokens=300)


class Stub:
    version = "stub"

    def __init__(self, tier: int = 1, confidence: float = 0.9) -> None:
        self.tier, self.confidence = tier, confidence

    def predict(self, prompt: str) -> Prediction:
        return Prediction(tier=self.tier, confidence=self.confidence, probabilities={self.tier: self.confidence},  # type: ignore[arg-type]
                          task_type="qa_classification", task_confidence=0.9)


@pytest.fixture
def catalog() -> ModelCatalog:
    return load_catalog()


def build(
    tmp_path: Path, catalog: ModelCatalog, world: World, *, tier: int = 1, confidence: float = 0.9, **settings: Any
) -> tuple[AutopilotService, Worker, FakeOpenAI]:
    fake = FakeOpenAI([world])
    client, _ = make_client(fake, max_attempts=1)
    routing = RoutingConfig(tiers={1: "local-llama", 2: "deepseek-flash", 3: "deepseek-v4-pro"},
                            confidence=ConfidenceRule(enabled=False, threshold=0.5))
    verification = VerificationSettings(**{"sample_rate_override": 1.0, **settings})
    repo = Repository(Database(tmp_path / "flow.db"))
    verifier = Verifier(client, catalog, routing, load_quality_config(catalog), verification, rng=random.Random(0))
    service = AutopilotService(router=Router(Stub(tier, confidence), routing, catalog), client=client, repo=repo,
                               verifier=verifier, settings=verification, rng=random.Random(0))
    return service, Worker(repo, verifier, verification), fake


ASK = [{"role": "user", "content": "What is the capital of France?"}]


def test_routing_text_uses_last_user_message() -> None:
    from autopilot.llm.schemas import ChatMessage

    msgs = [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="a"), ChatMessage(role="user", content="second")]
    assert routing_text(msgs) == "second"
    with pytest.raises(ValueError):
        routing_text([ChatMessage(role="system", content="s")])


async def test_async_pass_flow(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    service, worker, _ = build(tmp_path, catalog, World())
    result = await service.complete(ASK)
    assert result.output_text == "CHEAP-ANSWER" and result.served_model == "local-llama"
    assert result.verification_status == "queued"
    assert service.repo.queue_counts() == {"queued": 1}

    stats = await worker.run(once=True)
    assert (stats.processed, stats.passed) == (1, 1)
    row = service.repo.get_request(result.request_id)
    assert row is not None
    assert (row["verification_status"], row["judge_score"], row["quality_passed"]) == ("verified", 5, 1)
    assert row["verification_cost_usd"] > 0  # reference + judge are always counted
    assert service.repo.list_routing_failures() == []


async def test_async_failure_logs_escalation_and_feedback(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    world = World()
    world.routed_score = 2
    service, worker, _ = build(tmp_path, catalog, world)
    result = await service.complete(ASK)
    await worker.run(once=True)

    row = service.repo.get_request(result.request_id)
    assert row is not None and row["quality_passed"] == 0 and row["judge_score"] == 2
    assert row["escalated"] == 0  # async: logged only
    escalation = service.repo.list_escalations()[0]
    assert (escalation["mode"], escalation["original_model"], escalation["escalated_model"]) == ("async", "local-llama", "deepseek-v4-pro")
    assert escalation["quality_gap"] == 2 and escalation["answer_replaced"] == 0
    assert escalation["cost_delta_usd"] > 0
    failure = service.repo.list_routing_failures()[0]
    assert (failure.served_tier, failure.correct_tier, failure.prompt) == (1, 2, "What is the capital of France?")
    verification = service.repo.get_verification(result.request_id)
    assert verification is not None and verification["source"] == "sampled"


@pytest.mark.parametrize(
    ("tier", "sample_rate", "truncate", "expected"),
    [(1, 0.0, False, "not_sampled"), (3, 1.0, False, "skipped_top_tier"), (1, 1.0, True, "excluded_truncated")],
)
async def test_requests_that_are_not_queued(
    tmp_path: Path, catalog: ModelCatalog, api_key: str, tier: int, sample_rate: float, truncate: bool, expected: str
) -> None:
    world = World()
    if truncate:
        world.truncate.add("llama3.2:3b")
    service, _, _ = build(tmp_path, catalog, world, tier=tier, sample_rate_override=sample_rate)
    result = await service.complete(ASK)
    assert result.verification_status == expected
    assert service.repo.queue_counts() == {}
    assert service.repo.get_request(result.request_id)["truncated"] == int(truncate)  # type: ignore[index]


async def test_sync_escalation_replaces_failing_answer(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    world = World()
    world.routed_score = 1
    service, _, _ = build(tmp_path, catalog, world, confidence=0.3, escalation_mode="sync")
    result = await service.complete(ASK)

    assert result.escalated and result.output_text == "REFERENCE-ANSWER"
    assert result.served_model == "deepseek-v4-pro" and result.served_tier == 3
    assert result.verification is not None and result.verification.source == "sync"
    assert service.repo.queue_counts() == {}  # verified inline, not queued
    row = service.repo.get_request(result.request_id)
    assert row is not None and row["escalated"] == 1 and row["verification_status"] == "verified"
    escalation = service.repo.list_escalations()[0]
    assert escalation["mode"] == "sync" and escalation["answer_replaced"] == 1
    assert len(service.repo.list_routing_failures()) == 1


async def test_sync_mode_keeps_passing_answer_and_skips_confident(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    service, _, _ = build(tmp_path, catalog, World(), confidence=0.3, escalation_mode="sync")
    result = await service.complete(ASK)
    assert not result.escalated and result.output_text == "CHEAP-ANSWER" and result.verification_status == "verified"

    confident, _, _ = build(tmp_path / "b", catalog, World(), confidence=0.95, escalation_mode="sync")
    result = await confident.complete(ASK)
    assert result.verification is None and result.verification_status == "queued"


async def test_reference_truncated_or_unparseable_judge_is_excluded(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    world = World()
    world.truncate.add("deepseek-v4-pro")
    service, worker, _ = build(tmp_path, catalog, world)
    result = await service.complete(ASK)
    await worker.run(once=True)
    verification = service.repo.get_verification(result.request_id)
    assert verification is not None and verification["excluded_reason"] == "reference_truncated"
    assert verification["passed"] is None and verification["reference_cost_usd"] > 0
    assert service.repo.get_request(result.request_id)["verification_status"] == "excluded"  # type: ignore[index]

    world2 = World()
    world2.judge_reply = "not json"
    service2, worker2, _ = build(tmp_path / "b", catalog, world2)
    result2 = await service2.complete(ASK)
    await worker2.run(once=True)
    verification2 = service2.repo.get_verification(result2.request_id)
    assert verification2 is not None and verification2["excluded_reason"] == "judge_unparseable"
    assert verification2["judge_cost_usd"] > 0
    assert service2.repo.list_routing_failures() == []


async def test_failover_to_next_tier_and_total_failure(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    world = World()
    world.down.add("llama3.2:3b")
    service, _, _ = build(tmp_path, catalog, world)
    result = await service.complete(ASK)
    assert result.failover and result.served_model == "deepseek-flash" and result.output_text == "MID-ANSWER"
    row = service.repo.get_request(result.request_id)
    assert row is not None and (row["routed_model"], row["served_model"], row["served_tier"]) == ("local-llama", "deepseek-flash", 2)

    world.down |= {"deepseek-flash", "deepseek-v4-pro"}
    with pytest.raises(LLMRequestError):
        await service.complete(ASK)
    with service.repo.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests WHERE status = 'error'").fetchone()[0] == 1


async def test_worker_retries_then_fails_job(tmp_path: Path, catalog: ModelCatalog, api_key: str) -> None:
    world = World()
    service, worker, _ = build(tmp_path, catalog, world, job_max_attempts=1)
    result = await service.complete(ASK)
    world.down.add("deepseek-v4-pro")
    stats = await worker.run(once=True)
    assert stats.errors == 1
    assert service.repo.queue_counts() == {"failed": 1}
    assert service.repo.get_request(result.request_id)["verification_status"] == "failed"  # type: ignore[index]
