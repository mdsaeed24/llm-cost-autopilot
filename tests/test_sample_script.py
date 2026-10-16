"""scripts/sample_prompts.py end to end with a synthetic Dolly (no download)."""

from __future__ import annotations

import json
from pathlib import Path

from autopilot.dataset.io import read_candidates_csv, read_jsonl
from autopilot.dataset.schemas import LoadtestPrompt
from tests.conftest import load_script
from tests.test_sampling import CATEGORIES, synthetic_dolly


def run(tmp_path: Path, *extra: str) -> int:
    script = load_script("sample_prompts")
    return script.main(
        ["--data-dir", str(tmp_path), "--n-candidates", "60", "--loadtest-size", "100", *extra],
        loader=lambda _cache: synthetic_dolly(per_category=60),
    )


def test_writes_candidates_pool_and_manifest(tmp_path: Path) -> None:
    assert run(tmp_path) == 0

    candidates = read_candidates_csv(tmp_path / "candidates.csv")
    pool = read_jsonl(tmp_path / "loadtest_prompts.jsonl", LoadtestPrompt)
    manifest = json.loads((tmp_path / "raw" / "sample_manifest.json").read_text())

    assert len(candidates) == 60 and len(pool) == 100
    assert not {c.prompt for c in candidates} & {p.prompt for p in pool}
    assert not set(manifest["candidate_dolly_index"].values()) & set(manifest["loadtest_dolly_index"].values())
    assert manifest["license"] == "CC BY-SA 3.0"

    # Categories must never leak into the files the labeler sees.
    visible = (tmp_path / "candidates.csv").read_text() + (tmp_path / "loadtest_prompts.jsonl").read_text()
    for category in CATEGORIES:
        assert f",{category}" not in visible and f'"{category}"' not in visible


def test_same_seed_same_sample(tmp_path: Path) -> None:
    run(tmp_path / "a")
    run(tmp_path / "b")
    assert (tmp_path / "a" / "candidates.csv").read_text() == (tmp_path / "b" / "candidates.csv").read_text()


def test_refuses_to_resample_once_labeling_started(tmp_path: Path) -> None:
    run(tmp_path)
    (tmp_path / "labeled_prompts.jsonl").write_text('{"candidate_id": "c0001"}\n')
    assert run(tmp_path) == 1
    assert run(tmp_path, "--force") == 0
