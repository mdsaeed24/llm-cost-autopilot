"""SQLite connection handling and schema.

WAL ("write-ahead log") mode lets the API, the worker and the dashboard read while another
process writes — without it, readers would block on every write. Each operation opens its
own short-lived connection, which keeps things thread- and process-safe without a pool.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);

-- One row per API/CLI request.
CREATE TABLE IF NOT EXISTS requests (
    id                    TEXT PRIMARY KEY,
    created_at            TEXT NOT NULL,
    prompt_hash           TEXT NOT NULL,
    mode                  TEXT NOT NULL,              -- normal | evaluation
    predicted_tier        INTEGER,                    -- NULL if the classifier failed
    confidence            REAL,
    routed_tier           INTEGER NOT NULL,           -- after the low-confidence rule
    routed_up             INTEGER NOT NULL,
    classifier_version    TEXT,
    task_type             TEXT,
    task_confidence       REAL,
    routing_reason        TEXT,
    routed_model          TEXT NOT NULL,              -- model the router chose
    served_tier           INTEGER,                    -- tier whose answer was returned
    served_model          TEXT,                       -- differs after failover / sync escalation
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    serve_cost_usd        REAL NOT NULL DEFAULT 0,    -- calls made to answer the request
    verification_cost_usd REAL NOT NULL DEFAULT 0,    -- Tier-3 reference + judge calls
    latency_ms            REAL,
    finish_reason         TEXT,
    truncated             INTEGER NOT NULL DEFAULT 0,
    status                TEXT NOT NULL,              -- ok | error
    error                 TEXT,
    verification_status   TEXT NOT NULL,              -- see VerificationStatus
    verified              INTEGER NOT NULL DEFAULT 0,
    judge_score           INTEGER,                    -- agreement 1-5
    quality_passed        INTEGER,                    -- 1 / 0 / NULL (not judged)
    escalated             INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);

-- Background verification queue (instead of Redis).
CREATE TABLE IF NOT EXISTS verification_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL UNIQUE REFERENCES requests(id),
    status        TEXT NOT NULL,                      -- queued | running | done | failed
    attempts      INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL,                      -- JSON JobPayload
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    available_at  TEXT NOT NULL,                      -- not before (retry backoff)
    locked_at     TEXT,
    locked_by     TEXT,
    last_error    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON verification_jobs(status, available_at);

-- Result of comparing a routed answer with the Tier-3 reference.
CREATE TABLE IF NOT EXISTS verifications (
    request_id            TEXT PRIMARY KEY REFERENCES requests(id),
    created_at            TEXT NOT NULL,
    source                TEXT NOT NULL,              -- sampled (unbiased) | sync (low-confidence only)
    reference_model       TEXT NOT NULL,
    reference_output      TEXT,
    reference_cost_usd    REAL NOT NULL,
    reference_latency_ms  REAL,
    reference_truncated   INTEGER NOT NULL,
    judge_model           TEXT,
    judge_cost_usd        REAL NOT NULL DEFAULT 0,
    routed_score          INTEGER,
    reference_score       INTEGER,
    agreement             INTEGER,
    judge_reason          TEXT,
    rule                  TEXT,
    pass_threshold        INTEGER,
    passed                INTEGER,                    -- NULL when excluded
    excluded_reason       TEXT
);

-- Every escalation, sync (answer replaced) or async (logged only).
CREATE TABLE IF NOT EXISTS escalations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id         TEXT NOT NULL REFERENCES requests(id),
    created_at         TEXT NOT NULL,
    mode               TEXT NOT NULL,                 -- sync | async
    original_model     TEXT NOT NULL,
    escalated_model    TEXT NOT NULL,
    original_cost_usd  REAL NOT NULL,
    escalated_cost_usd REAL NOT NULL,
    cost_delta_usd     REAL NOT NULL,
    judge_score        INTEGER,
    quality_gap        INTEGER,                       -- pass_threshold - agreement (>0 = below the bar)
    answer_replaced    INTEGER NOT NULL
);

-- Confirmed routing failures: training examples for the feedback loop.
CREATE TABLE IF NOT EXISTS routing_failures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL UNIQUE REFERENCES requests(id),
    created_at      TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    served_tier     INTEGER NOT NULL,
    correct_tier    INTEGER NOT NULL,                 -- one tier up
    task_type       TEXT,
    judge_score     INTEGER,
    judge_reason    TEXT,
    used_in_version TEXT                              -- classifier version that first trained on it
);

-- Runtime routing-config changes (PUT /v1/routing-config). The latest row overrides
-- config/routing.yaml for both the API and the worker, and survives restarts.
CREATE TABLE IF NOT EXISTS config_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    actor       TEXT,
    old_config  TEXT NOT NULL,                        -- JSON RoutingConfig
    new_config  TEXT NOT NULL,
    reason      TEXT
);
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    """Fixed-width UTC timestamp, so string comparison in SQL equals time comparison."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Database:
    """A SQLite file with the project schema."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived autocommit connection with rows as dict-like objects."""
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """All-or-nothing block. ``immediate`` takes the write lock up front (used to claim jobs)."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            if conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 0:
                conn.execute("INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,))
