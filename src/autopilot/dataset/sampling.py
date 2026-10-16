"""Turn Databricks Dolly 15k into labeling candidates and a separate load-test pool.

Dolly's ``category`` column is used ONLY here, to stratify the sample so hard prompts are
well represented. It is never written to the candidates file or shown to the labeler,
because seeing "creative_writing" next to a prompt would anchor the human's tier label.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from autopilot.dataset.schemas import Candidate, LoadtestPrompt

DOLLY_DATASET = "databricks/databricks-dolly-15k"
DOLLY_LICENSE = "CC BY-SA 3.0"

# open_qa / general_qa questions with these cues tend to need explanation or judgment,
# so they form their own "open_reasoning" stratum.
REASONING_CUES = re.compile(
    r"\b(why|explain|compare|contrast|pros and cons|advantages|disadvantages|should|"
    r"what if|how would|how could|argue|justify|evaluate)\b",
    re.IGNORECASE,
)

# Strata expected to yield mostly Tier-3 labels. 120 of 300 candidates (40%) come from
# these so that, even if many end up labeled Tier 2, there are still 40+ Tier-3 examples.
TIER3_LIKELY_STRATA = ("creative_writing", "brainstorming", "open_reasoning")

DEFAULT_QUOTAS: dict[str, int] = {
    "creative_writing": 45,
    "brainstorming": 40,
    "open_reasoning": 35,
    "open_qa": 25,
    "general_qa": 20,
    "closed_qa": 35,
    "information_extraction": 35,
    "summarization": 35,
    "classification": 30,
}


@dataclass(frozen=True)
class DollyRecord:
    """One Dolly row (the reference ``response`` is not needed)."""

    index: int
    instruction: str
    context: str
    category: str

    @property
    def prompt(self) -> str:
        return build_prompt(self.instruction, self.context)

    @property
    def stratum(self) -> str:
        return stratum_of(self)


def build_prompt(instruction: str, context: str) -> str:
    """Instruction first, then the context (when present) under a ``Context:`` header."""
    instruction, context = instruction.strip(), context.strip()
    return f"{instruction}\n\nContext:\n{context}" if context else instruction


def stratum_of(record: DollyRecord) -> str:
    """Dolly category, except open-ended QA with reasoning cues gets its own stratum."""
    if record.category in ("open_qa", "general_qa") and REASONING_CUES.search(record.instruction):
        return "open_reasoning"
    return record.category


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def clean_records(records: Iterable[DollyRecord], *, min_chars: int, max_chars: int) -> list[DollyRecord]:
    """Drop prompts that are too short/long and exact duplicates (case/whitespace-insensitive)."""
    seen: set[str] = set()
    kept: list[DollyRecord] = []
    for record in records:
        prompt = record.prompt
        if not (min_chars <= len(prompt) <= max_chars):
            continue
        key = _normalise(prompt)
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept


def scale_quotas(quotas: Mapping[str, int], total: int) -> dict[str, int]:
    """Rescale quotas to sum to ``total``, keeping proportions (largest-remainder rounding)."""
    base = sum(quotas.values())
    exact = {k: v * total / base for k, v in quotas.items()}
    scaled = {k: int(v) for k, v in exact.items()}
    leftover = total - sum(scaled.values())
    for key in sorted(exact, key=lambda k: exact[k] - scaled[k], reverse=True)[:leftover]:
        scaled[key] += 1
    return scaled


def stratified_sample(
    records: Sequence[DollyRecord], quotas: Mapping[str, int], rng: random.Random
) -> tuple[list[DollyRecord], dict[str, int]]:
    """Sample ``quotas[stratum]`` records per stratum, then shuffle.

    Returns the sample and any shortfall per stratum (when a stratum has too few records).
    """
    by_stratum: dict[str, list[DollyRecord]] = defaultdict(list)
    for record in records:
        by_stratum[record.stratum].append(record)

    sample: list[DollyRecord] = []
    shortfall: dict[str, int] = {}
    for stratum in sorted(quotas):  # sorted: same seed → same sample regardless of dict order
        pool = by_stratum.get(stratum, [])
        want = quotas[stratum]
        if len(pool) < want:
            shortfall[stratum] = want - len(pool)
        sample.extend(rng.sample(pool, min(want, len(pool))))
    rng.shuffle(sample)  # interleave strata so labeling order gives no hint
    return sample, shortfall


def sample_pool(
    records: Sequence[DollyRecord], *, exclude: set[int], size: int, rng: random.Random
) -> list[DollyRecord]:
    """Uniform random sample (NOT stratified) of records whose index is not in ``exclude``.

    The load-test pool keeps Dolly's natural mix, so it looks like real traffic rather than
    our hard-prompt-heavy training sample.
    """
    available = [r for r in records if r.index not in exclude]
    if len(available) < size:
        raise ValueError(f"Only {len(available)} records available for a pool of {size}")
    return rng.sample(available, size)


def to_candidates(records: Sequence[DollyRecord]) -> list[Candidate]:
    """Assign ids c0001, c0002, ... in the (already shuffled) order."""
    return [Candidate(candidate_id=f"c{i:04d}", prompt=r.prompt) for i, r in enumerate(records, start=1)]


def to_loadtest(records: Sequence[DollyRecord]) -> list[LoadtestPrompt]:
    """Assign ids lt0001, lt0002, ..."""
    return [LoadtestPrompt(loadtest_id=f"lt{i:04d}", prompt=r.prompt) for i, r in enumerate(records, start=1)]


def load_dolly(cache_dir: Path) -> list[DollyRecord]:
    """Download (once, cached) and load Dolly 15k from Hugging Face."""
    from datasets import load_dataset  # heavy import; only needed here

    dataset = load_dataset(DOLLY_DATASET, split="train", cache_dir=str(cache_dir))
    return [
        DollyRecord(index=i, instruction=row["instruction"], context=row["context"] or "", category=row["category"])
        for i, row in enumerate(dataset)
    ]
