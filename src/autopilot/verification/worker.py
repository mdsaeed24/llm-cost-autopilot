"""Background worker: claims verification jobs from SQLite and processes them concurrently."""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from autopilot.llm.client import LLMRequestError
from autopilot.routing.config import RoutingConfig
from autopilot.settings import VerificationSettings
from autopilot.storage.records import Job
from autopilot.storage.repository import Repository
from autopilot.verification.verifier import Verifier, consequences

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    processed: int = 0
    passed: int = 0
    failed_quality: int = 0
    excluded: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    reasons: list[str] = field(default_factory=list)


class Worker:
    """Pulls jobs in batches of ``worker_concurrency`` and verifies them in parallel."""

    def __init__(
        self,
        repo: Repository,
        verifier: Verifier,
        settings: VerificationSettings,
        *,
        refresh_routing: Callable[[], RoutingConfig] | None = None,
    ) -> None:
        self.repo = repo
        self.verifier = verifier
        self.settings = settings
        self.refresh_routing = refresh_routing  # picks up runtime routing changes made via the API
        self.worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.stats = WorkerStats()

    async def process(self, job: Job) -> None:
        try:
            outcome = await self.verifier.verify(job.request_id, job.payload, source="sampled")
        except LLMRequestError as error:
            retry = self.repo.fail_job(job, str(error), max_attempts=self.settings.job_max_attempts)
            self.stats.errors += 1
            logger.warning("Job %d failed (attempt %d, retry=%s): %s", job.id, job.attempts, retry, error)
            return
        escalation, failure = consequences(job.request_id, job.payload, outcome, mode="async")
        self.repo.save_verification(outcome.record, escalation=escalation, failure=failure)
        self.repo.complete_job(job.id)

        self.stats.processed += 1
        self.stats.cost_usd += outcome.record.total_cost_usd
        if outcome.record.passed is None:
            self.stats.excluded += 1
        elif outcome.record.passed:
            self.stats.passed += 1
        else:
            self.stats.failed_quality += 1
            self.stats.reasons.append(outcome.record.judge_reason or "")

    async def run_batch(self) -> int:
        """Claim and process one batch. Returns how many jobs were claimed."""
        if self.refresh_routing is not None:
            self.verifier.routing = self.refresh_routing()
        jobs = self.repo.claim_jobs(
            self.worker_id,
            limit=self.settings.worker_concurrency,
            max_attempts=self.settings.job_max_attempts,
            visibility_timeout_s=self.settings.job_visibility_timeout_s,
        )
        if jobs:
            await asyncio.gather(*(self.process(job) for job in jobs))
        return len(jobs)

    async def run(self, *, once: bool = False, limit: int | None = None, stop: asyncio.Event | None = None) -> WorkerStats:
        """Process jobs until the queue is empty (``once``), ``limit`` jobs are done, or ``stop`` is set."""
        stop = stop or asyncio.Event()
        claimed_total = 0
        while not stop.is_set():
            if limit is not None and claimed_total >= limit:
                break
            claimed = await self.run_batch()
            claimed_total += claimed
            if claimed == 0:
                if once:
                    break
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.settings.worker_poll_interval_s)
                except TimeoutError:
                    pass
        return self.stats
