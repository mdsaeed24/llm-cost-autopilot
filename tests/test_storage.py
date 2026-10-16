"""SQLite storage: schema, request log, and the job queue's claim/retry/recovery semantics."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from autopilot.llm.schemas import ChatMessage
from autopilot.storage.db import Database, utc_now
from autopilot.storage.records import (
    EscalationRecord,
    JobPayload,
    RequestRecord,
    RoutingFailureRecord,
    VerificationRecord,
)
from autopilot.storage.repository import Repository


def request(rid: str, **overrides: object) -> RequestRecord:
    values: dict[str, object] = dict(
        id=rid, created_at=utc_now(), prompt_hash="h", mode="normal", predicted_tier=1, confidence=0.9,
        routed_tier=1, routed_up=False, classifier_version="v2", task_type="qa_classification",
        task_confidence=0.8, routing_reason="r", routed_model="local-llama", served_tier=1,
        served_model="local-llama", serve_cost_usd=0.0, verification_status="queued",
    )
    values.update(overrides)
    return RequestRecord.model_validate(values)


PAYLOAD = JobPayload(
    messages=[ChatMessage(role="user", content="hi")], prompt="hi", routed_output="hello",
    served_model="local-llama", served_tier=1, served_cost_usd=0.0, task_type="qa_classification", task_confidence=0.8,
)


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(Database(tmp_path / "test.db"))


def test_schema_and_wal(repo: Repository) -> None:
    with repo.db.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"requests", "verification_jobs", "verifications", "escalations", "routing_failures"} <= tables
    Database(repo.db.path)  # re-opening is idempotent


def test_request_and_job_are_written_together(repo: Repository) -> None:
    repo.insert_request(request("r1"), job=PAYLOAD)
    repo.insert_request(request("r2", verification_status="not_sampled"))
    assert repo.get_request("r1")["routed_model"] == "local-llama"  # type: ignore[index]
    assert repo.queue_counts() == {"queued": 1}


def test_failed_request_insert_rolls_back_job(repo: Repository) -> None:
    repo.insert_request(request("r1"), job=PAYLOAD)
    with pytest.raises(Exception):
        repo.insert_request(request("r1"), job=PAYLOAD)  # duplicate id
    assert repo.queue_counts() == {"queued": 1}


def test_claims_are_exclusive_and_ordered(repo: Repository) -> None:
    for i in range(5):
        repo.insert_request(request(f"r{i}"), job=PAYLOAD)
    first = repo.claim_jobs("w1", limit=3, max_attempts=3, visibility_timeout_s=600)
    second = repo.claim_jobs("w2", limit=3, max_attempts=3, visibility_timeout_s=600)
    assert [j.request_id for j in first] == ["r0", "r1", "r2"]
    assert [j.request_id for j in second] == ["r3", "r4"]
    assert all(j.attempts == 1 for j in first + second)
    assert repo.claim_jobs("w3", limit=3, max_attempts=3, visibility_timeout_s=600) == []
    repo.complete_job(first[0].id)
    assert repo.queue_counts() == {"done": 1, "running": 4}


def test_failed_jobs_back_off_then_give_up(repo: Repository) -> None:
    repo.insert_request(request("r1"), job=PAYLOAD)
    now = utc_now()
    job = repo.claim_jobs("w", limit=1, max_attempts=2, visibility_timeout_s=600, now=now)[0]
    assert repo.fail_job(job, "boom", max_attempts=2, now=now) is True
    assert repo.claim_jobs("w", limit=1, max_attempts=2, visibility_timeout_s=600, now=now) == []  # backing off
    later = now + timedelta(minutes=5)
    job = repo.claim_jobs("w", limit=1, max_attempts=2, visibility_timeout_s=600, now=later)[0]
    assert job.attempts == 2
    assert repo.fail_job(job, "boom again", max_attempts=2, now=later) is False
    assert repo.queue_counts() == {"failed": 1}
    assert repo.get_request("r1")["verification_status"] == "failed"  # type: ignore[index]


def test_jobs_from_dead_workers_are_recovered(repo: Repository) -> None:
    repo.insert_request(request("r1"), job=PAYLOAD)
    now = utc_now()
    repo.claim_jobs("dead-worker", limit=1, max_attempts=3, visibility_timeout_s=60, now=now)
    assert repo.claim_jobs("w2", limit=1, max_attempts=3, visibility_timeout_s=60, now=now) == []
    recovered = repo.claim_jobs("w2", limit=1, max_attempts=3, visibility_timeout_s=60, now=now + timedelta(seconds=120))
    assert [j.request_id for j in recovered] == ["r1"]
    assert recovered[0].attempts == 2


def test_save_verification_updates_request_and_logs_consequences(repo: Repository) -> None:
    repo.insert_request(request("r1"), job=PAYLOAD)
    now = utc_now()
    verification = VerificationRecord(
        request_id="r1", created_at=now, source="sampled", reference_model="deepseek-v4-pro",
        reference_output="ref", reference_cost_usd=0.002, reference_latency_ms=900, reference_truncated=False,
        judge_model="deepseek-v4-pro", judge_cost_usd=0.001, routed_score=2, reference_score=5, agreement=2,
        judge_reason="missed the point", rule="qa_classification", pass_threshold=4, passed=False,
    )
    escalation = EscalationRecord(
        request_id="r1", created_at=now, mode="async", original_model="local-llama",
        escalated_model="deepseek-v4-pro", original_cost_usd=0, escalated_cost_usd=0.002, cost_delta_usd=0.002,
        judge_score=2, quality_gap=2, answer_replaced=False,
    )
    failure = RoutingFailureRecord(
        request_id="r1", created_at=now, prompt="hi", served_tier=1, correct_tier=2,
        task_type="qa_classification", judge_score=2, judge_reason="missed the point",
    )
    repo.save_verification(verification, escalation=escalation, failure=failure)

    row = repo.get_request("r1")
    assert row is not None
    assert (row["verification_status"], row["verified"], row["judge_score"], row["quality_passed"]) == ("verified", 1, 2, 0)
    assert row["verification_cost_usd"] == pytest.approx(0.003)
    assert row["escalated"] == 0  # async: the user kept the routed answer
    assert repo.list_escalations()[0]["cost_delta_usd"] == pytest.approx(0.002)
    failures = repo.list_routing_failures()
    assert [(f.served_tier, f.correct_tier) for f in failures] == [(1, 2)]

    repo.mark_failures_used([failures[0].id], "v3")  # type: ignore[list-item]
    assert repo.list_routing_failures()[0].used_in_version == "v3"
    summary = repo.verification_summary()
    assert (summary["verified"], summary["failed"], summary["routing_failures"]) == (1, 1, 1)
