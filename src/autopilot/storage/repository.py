"""All reads and writes, including the SQLite-backed job queue.

Queue design: a job is *claimed* with a single ``UPDATE ... RETURNING`` inside an
``IMMEDIATE`` transaction, so two workers can never claim the same job. A job whose worker
died (status ``running`` for longer than the visibility timeout) is put back in the queue.
Failed attempts are retried with exponential backoff until ``max_attempts``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel

from autopilot.storage.db import Database, iso, utc_now
from autopilot.storage.records import (
    EscalationRecord,
    Job,
    JobPayload,
    RequestRecord,
    RoutingFailureRecord,
    VerificationRecord,
)

RETRY_BASE_DELAY_S = 30.0


def _row(model: BaseModel) -> dict[str, Any]:
    """Model → column dict (datetimes as ISO strings, bools as 0/1)."""
    out: dict[str, Any] = {}
    for key, value in model.model_dump().items():
        if isinstance(value, datetime):
            value = iso(value)
        elif isinstance(value, bool):
            value = int(value)
        out[key] = value
    return out


def _insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> int:
    columns = ", ".join(values)
    placeholders = ", ".join(f":{k}" for k in values)
    cursor = conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
    return int(cursor.lastrowid or 0)


class Repository:
    """Data access for requests, the verification queue, and verification results."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------- requests ----------

    def insert_request(self, request: RequestRecord, *, job: JobPayload | None = None) -> None:
        """Log a request, and enqueue its verification job in the same transaction."""
        with self.db.transaction() as conn:
            _insert(conn, "requests", _row(request))
            if job is not None:
                now = iso(utc_now())
                conn.execute(
                    "INSERT INTO verification_jobs (request_id, status, payload, created_at, updated_at, available_at)"
                    " VALUES (?, 'queued', ?, ?, ?, ?)",
                    (request.id, job.model_dump_json(), now, now, now),
                )

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        return dict(row) if row else None

    # ---------- queue ----------

    def claim_jobs(
        self,
        worker_id: str,
        *,
        limit: int,
        max_attempts: int,
        visibility_timeout_s: float,
        now: datetime | None = None,
    ) -> list[Job]:
        """Atomically take up to ``limit`` ready jobs (also recovering jobs from dead workers)."""
        now = now or utc_now()
        stale_before = iso(now - timedelta(seconds=visibility_timeout_s))
        with self.db.transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE verification_jobs SET status = CASE WHEN attempts >= ? THEN 'failed' ELSE 'queued' END,"
                " locked_at = NULL, locked_by = NULL, last_error = 'worker timed out', updated_at = ?"
                " WHERE status = 'running' AND locked_at < ?",
                (max_attempts, iso(now), stale_before),
            )
            self._mark_failed_requests(conn)
            rows = conn.execute(
                "UPDATE verification_jobs SET status = 'running', attempts = attempts + 1,"
                " locked_at = ?, locked_by = ?, updated_at = ?"
                " WHERE id IN (SELECT id FROM verification_jobs WHERE status = 'queued' AND available_at <= ?"
                "              ORDER BY id LIMIT ?)"
                " RETURNING id, request_id, attempts, payload",
                (iso(now), worker_id, iso(now), iso(now), limit),
            ).fetchall()
        jobs = [
            Job(id=r["id"], request_id=r["request_id"], attempts=r["attempts"],
                payload=JobPayload.model_validate_json(r["payload"]))
            for r in rows
        ]
        return sorted(jobs, key=lambda j: j.id)

    def complete_job(self, job_id: int) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE verification_jobs SET status = 'done', locked_at = NULL, updated_at = ? WHERE id = ?",
                (iso(utc_now()), job_id),
            )

    def fail_job(self, job: Job, error: str, *, max_attempts: int, now: datetime | None = None) -> bool:
        """Record a failed attempt. Returns True if the job will be retried."""
        now = now or utc_now()
        retry = job.attempts < max_attempts
        delay = RETRY_BASE_DELAY_S * 2 ** (job.attempts - 1)
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE verification_jobs SET status = ?, last_error = ?, locked_at = NULL, locked_by = NULL,"
                " available_at = ?, updated_at = ? WHERE id = ?",
                ("queued" if retry else "failed", error[:2000], iso(now + timedelta(seconds=delay)), iso(now), job.id),
            )
            self._mark_failed_requests(conn)
        return retry

    @staticmethod
    def _mark_failed_requests(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE requests SET verification_status = 'failed' WHERE verification_status = 'queued'"
            " AND id IN (SELECT request_id FROM verification_jobs WHERE status = 'failed')"
        )

    def queue_counts(self) -> dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM verification_jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ---------- verification results ----------

    def save_verification(
        self,
        verification: VerificationRecord,
        *,
        escalation: EscalationRecord | None = None,
        failure: RoutingFailureRecord | None = None,
    ) -> None:
        """Store a verification and its consequences, and update the request row — atomically."""
        judged = verification.passed is not None
        with self.db.transaction() as conn:
            _insert(conn, "verifications", _row(verification))
            conn.execute(
                "UPDATE requests SET verification_status = ?, verified = ?, judge_score = ?, quality_passed = ?,"
                " verification_cost_usd = verification_cost_usd + ?, escalated = MAX(escalated, ?) WHERE id = ?",
                (
                    "verified" if judged else "excluded",
                    int(judged),
                    verification.agreement,
                    None if verification.passed is None else int(verification.passed),
                    verification.total_cost_usd,
                    int(escalation is not None and escalation.answer_replaced),
                    verification.request_id,
                ),
            )
            if escalation is not None:
                _insert(conn, "escalations", _row(escalation))
            if failure is not None:
                values = _row(failure)
                values.pop("id")
                _insert(conn, "routing_failures", values)

    def get_verification(self, request_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM verifications WHERE request_id = ?", (request_id,)).fetchone()
        return dict(row) if row else None

    def list_escalations(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM escalations ORDER BY id")]

    # ---------- feedback loop ----------

    def list_routing_failures(self) -> list[RoutingFailureRecord]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM routing_failures ORDER BY id").fetchall()
        return [RoutingFailureRecord.model_validate(dict(r)) for r in rows]

    def mark_failures_used(self, failure_ids: Iterable[int], version: str) -> None:
        with self.db.connect() as conn:
            conn.executemany(
                "UPDATE routing_failures SET used_in_version = ? WHERE id = ? AND used_in_version IS NULL",
                [(version, i) for i in failure_ids],
            )

    # ---------- routing config changes ----------

    def record_config_change(self, old: str, new: str, *, actor: str | None, reason: str | None) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO config_changes (created_at, actor, old_config, new_config, reason) VALUES (?, ?, ?, ?, ?)",
                (iso(utc_now()), actor, old, new, reason),
            )
            return int(cursor.lastrowid or 0)

    def latest_config(self) -> str | None:
        """JSON of the most recent runtime routing config, if any."""
        with self.db.connect() as conn:
            row = conn.execute("SELECT new_config FROM config_changes ORDER BY id DESC LIMIT 1").fetchone()
        return row["new_config"] if row else None

    def list_config_changes(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM config_changes ORDER BY id")]

    # ---------- summaries ----------

    def verification_summary(self) -> dict[str, Any]:
        """Quick numbers for CLI output (the full metrics live in Phase 4)."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS requests,"
                " SUM(serve_cost_usd) AS serve_cost, SUM(verification_cost_usd) AS verification_cost,"
                " SUM(verification_status = 'verified') AS verified,"
                " SUM(quality_passed = 1) AS passed, SUM(quality_passed = 0) AS failed,"
                " SUM(verification_status IN ('excluded', 'excluded_truncated')) AS excluded,"
                " SUM(truncated) AS truncated, SUM(escalated) AS escalated_sync"
                " FROM requests"
            ).fetchone()
            failures = conn.execute("SELECT COUNT(*) FROM routing_failures").fetchone()[0]
        summary = {k: (row[k] or 0) for k in row.keys()}
        summary["routing_failures"] = failures
        return summary
