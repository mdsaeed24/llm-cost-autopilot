"""Decide whether a request gets a background verification.

The sample must be random and independent of anything the router knows (confidence, tier,
task type) — otherwise the measured quality parity would be biased. Low-confidence requests
checked by sync escalation are stored with ``source="sync"`` and kept out of parity numbers.
"""

from __future__ import annotations

import random

from autopilot.storage.records import VerificationStatus

TOP_TIER = 3


def plan_verification(
    *, served_tier: int, truncated: bool, sample_rate: float, rng: random.Random
) -> VerificationStatus:
    """``queued`` if this request should be verified in the background, else why not."""
    if served_tier >= TOP_TIER:
        return "skipped_top_tier"
    if truncated:
        return "excluded_truncated"
    return "queued" if rng.random() < sample_rate else "not_sampled"
