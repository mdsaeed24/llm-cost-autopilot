"""Dolly sampling: prompt building, stratification, de-duplication, pool separation, file I/O."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from autopilot.dataset.io import DataFileError, read_candidates_csv, write_candidates_csv
from autopilot.dataset.sampling import (
    DEFAULT_QUOTAS,
    TIER3_LIKELY_STRATA,
    DollyRecord,
    build_prompt,
    clean_records,
    sample_pool,
    scale_quotas,
    stratified_sample,
    stratum_of,
    to_candidates,
)
from autopilot.dataset.schemas import Candidate

CATEGORIES = (
    "open_qa", "general_qa", "classification", "closed_qa",
    "brainstorming", "information_extraction", "summarization", "creative_writing",
)


def synthetic_dolly(per_category: int = 60) -> list[DollyRecord]:
    """Fake Dolly rows: every category, some with reasoning cues, some with context."""
    records: list[DollyRecord] = []
    for category in CATEGORIES:
        for i in range(per_category):
            question = f"Why does {category} item {i} work?" if i % 3 == 0 else f"Tell me about {category} item {i}"
            context = f"Some context for {category} {i}." if i % 2 else ""
            records.append(DollyRecord(index=len(records), instruction=question, context=context, category=category))
    return records


def test_build_prompt_with_and_without_context() -> None:
    assert build_prompt("  Summarise this.  ", "") == "Summarise this."
    assert build_prompt("Summarise this.", "  Long text.\n") == "Summarise this.\n\nContext:\nLong text."


@pytest.mark.parametrize(
    ("category", "instruction", "expected"),
    [
        ("open_qa", "Why is the sky blue?", "open_reasoning"),
        ("general_qa", "Should I buy or rent a house?", "open_reasoning"),
        ("general_qa", "What are the pros and cons of remote work?", "open_reasoning"),
        ("open_qa", "Who wrote Hamlet?", "open_qa"),
        ("closed_qa", "Why did the company fail?", "closed_qa"),  # only open-ended QA is re-bucketed
        ("creative_writing", "Write a poem", "creative_writing"),
    ],
)
def test_stratum_of(category: str, instruction: str, expected: str) -> None:
    assert stratum_of(DollyRecord(0, instruction, "", category)) == expected


def test_clean_records_filters_length_and_duplicates() -> None:
    records = [
        DollyRecord(0, "Hi", "", "open_qa"),                              # too short
        DollyRecord(1, "What is a cat?", "", "open_qa"),
        DollyRecord(2, "  what is a   CAT? ", "", "general_qa"),           # duplicate after normalising
        DollyRecord(3, "Summarise", "x" * 100, "summarization"),           # too long
        DollyRecord(4, "Summarise", "short text", "summarization"),
    ]
    kept = clean_records(records, min_chars=10, max_chars=60)
    assert [r.index for r in kept] == [1, 4]


def test_scale_quotas_preserves_total_and_proportions() -> None:
    assert sum(scale_quotas(DEFAULT_QUOTAS, 300).values()) == 300
    assert scale_quotas(DEFAULT_QUOTAS, 300) == DEFAULT_QUOTAS
    half = scale_quotas(DEFAULT_QUOTAS, 150)
    assert sum(half.values()) == 150
    assert half["creative_writing"] in (22, 23)


def test_default_quotas_favour_hard_prompts() -> None:
    hard = sum(DEFAULT_QUOTAS[s] for s in TIER3_LIKELY_STRATA)
    assert hard >= 0.4 * sum(DEFAULT_QUOTAS.values())


def test_stratified_sample_meets_quotas_and_is_reproducible() -> None:
    records = synthetic_dolly()
    quotas = {"creative_writing": 10, "open_reasoning": 8, "closed_qa": 5}
    sample, shortfall = stratified_sample(records, quotas, random.Random(7))
    again, _ = stratified_sample(records, quotas, random.Random(7))
    assert shortfall == {}
    assert len(sample) == 23
    assert [r.index for r in sample] == [r.index for r in again]
    counts = {s: sum(r.stratum == s for r in sample) for s in quotas}
    assert counts == quotas
    assert len({r.index for r in sample}) == 23  # no duplicates


def test_stratified_sample_reports_shortfall() -> None:
    records = synthetic_dolly(per_category=3)
    sample, shortfall = stratified_sample(records, {"creative_writing": 10}, random.Random(0))
    assert len(sample) == 3
    assert shortfall == {"creative_writing": 7}


def test_pool_excludes_candidates() -> None:
    records = synthetic_dolly()
    sample, _ = stratified_sample(records, {"creative_writing": 20, "brainstorming": 20}, random.Random(1))
    exclude = {r.index for r in sample}
    pool = sample_pool(records, exclude=exclude, size=100, rng=random.Random(1))
    assert len(pool) == 100
    assert not exclude & {r.index for r in pool}


def test_pool_too_small_raises() -> None:
    with pytest.raises(ValueError, match="Only"):
        sample_pool(synthetic_dolly(per_category=1), exclude=set(), size=100, rng=random.Random(0))


def test_candidates_have_ids_and_no_category(tmp_path: Path) -> None:
    records = synthetic_dolly(per_category=2)
    candidates = to_candidates(records)
    assert [c.candidate_id for c in candidates[:3]] == ["c0001", "c0002", "c0003"]
    assert set(Candidate.model_fields) == {"candidate_id", "prompt"}

    path = tmp_path / "candidates.csv"
    write_candidates_csv(path, candidates)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "candidate_id,prompt"
    for category in CATEGORIES:
        assert category not in header


def test_csv_round_trips_tricky_text(tmp_path: Path) -> None:
    tricky = [
        Candidate(candidate_id="c0001", prompt='Line one, with "quotes"\n\nContext:\nline, two\r\nthree'),
        Candidate(candidate_id="c0002", prompt="Unicode — café ✓ [bold]not markup[/bold]"),
    ]
    path = tmp_path / "candidates.csv"
    write_candidates_csv(path, tricky)
    assert read_candidates_csv(path) == tricky


def test_csv_with_wrong_columns_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text("candidate_id,prompt,category\nc1,hello,open_qa\n", encoding="utf-8")
    with pytest.raises(DataFileError, match="expected columns"):
        read_candidates_csv(path)
