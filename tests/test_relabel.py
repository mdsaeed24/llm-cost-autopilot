"""Labeling round 2: queue order, self-agreement, and the --round 2 script flow."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from autopilot.classifier.training import TestSplit
from autopilot.dataset.io import read_jsonl, write_candidates_csv, write_jsonl
from autopilot.dataset.relabel import agreement, round2_queue
from autopilot.dataset.schemas import RUBRIC_VERSION, Candidate, LabeledPrompt, rubric_text
from tests.conftest import load_script
from tests.test_labeling_session import Keyboard

CANDIDATES = [Candidate(candidate_id=f"c{i:02d}", prompt=f"Prompt {i}") for i in range(1, 11)]


def label(cid: str, tier: int, *, rubric: str | None = None) -> LabeledPrompt:
    return LabeledPrompt(candidate_id=cid, prompt=f"p {cid}", tier=tier, task_type="qa_classification", rubric=rubric)  # type: ignore[arg-type]


def test_queue_puts_test_prompts_first_then_old_then_new() -> None:
    previous = [label(f"c{i:02d}", 1) for i in range(1, 7)]  # c01..c06 labeled in round 1
    test_ids = {"c02", "c05"}
    queue = [c.candidate_id for c in round2_queue(CANDIDATES, previous, test_ids)]
    assert set(queue[:2]) == test_ids
    assert set(queue[2:6]) == {"c01", "c03", "c04", "c06"}
    assert set(queue[6:]) == {"c07", "c08", "c09", "c10"}
    assert queue == [c.candidate_id for c in round2_queue(CANDIDATES, previous, test_ids)]  # deterministic


def test_agreement_counts() -> None:
    previous = [label("a", 1), label("b", 2), label("c", 3), label("d", 1)]
    current = [label("a", 1), label("b", 3), label("c", 1), label("x", 2)]  # "x" wasn't in round 1
    result = agreement(previous, current)
    assert (result.n, result.same, result.moved_up, result.moved_down) == (3, 1, 1, 1)
    assert result.rate == 1 / 3
    assert result.confusion == [[1, 0, 0], [0, 0, 1], [1, 0, 0]]


def test_agreement_empty() -> None:
    assert agreement([label("a", 1)], []).n == 0


def test_rubric_text_lists_rules_in_order() -> None:
    text = rubric_text()
    assert text.splitlines()[0].startswith("1. The answer is already in the given text")
    assert "LOWER" in text.splitlines()[-1]


def test_round2_script_writes_separate_file_and_reports_agreement(tmp_path: Path) -> None:
    write_candidates_csv(tmp_path / "candidates.csv", CANDIDATES)
    round1 = [label("c01", 3), label("c02", 1)]
    write_jsonl(tmp_path / "labeled_prompts.jsonl", round1)
    (tmp_path / "test_split.json").write_text(
        TestSplit(created_at="2026-01-01T00:00:00Z", seed=1, test_size=0.2, test_ids=["c02"]).model_dump_json()  # type: ignore[arg-type]
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=160)
    script = load_script("label_prompts")

    # First prompt shown must be the test prompt c02; label it 1, then c01 as 2, then quit.
    code = script.main(["--data-dir", str(tmp_path), "--round", "2"], ask=Keyboard("1", "2", "", "2", "2", "", "q"), console=console)

    assert code == 0
    out = buffer.getvalue()
    assert "Rules (rules-v2)" in out and "first match decides" in out
    assert out.index("c02") < out.index("c01")
    v2 = read_jsonl(tmp_path / "labeled_prompts_v2.jsonl", LabeledPrompt)
    assert [(r.candidate_id, r.tier, r.rubric) for r in v2] == [("c02", 1, RUBRIC_VERSION), ("c01", 2, RUBRIC_VERSION)]
    assert read_jsonl(tmp_path / "labeled_prompts.jsonl", LabeledPrompt) == round1  # round 1 untouched
    assert "Agreement with your round-1 labels: 50%" in out


# ---------- round 3: memory-only fact review (rules-v3) ----------

from autopilot.dataset.relabel import latest_labels_file, merge_review, needs_fact_review, round3_queue  # noqa: E402
from autopilot.dataset.schemas import LATEST_RUBRIC  # noqa: E402

V2 = [
    LabeledPrompt(candidate_id="m1", prompt="Who wrote Hamlet?", tier=1, task_type="qa_classification", rubric="rules-v2"),
    LabeledPrompt(candidate_id="m2", prompt="When did WW2 end?", tier=1, task_type="qa_classification", rubric="rules-v2"),
    LabeledPrompt(candidate_id="c1", prompt="Who is named?\n\nContext:\nAda Lovelace wrote it.", tier=1,
                  task_type="extraction_reformatting", rubric="rules-v2"),
    LabeledPrompt(candidate_id="h1", prompt="Why do we dream?", tier=3, task_type="reasoning_creative", rubric="rules-v2"),
    LabeledPrompt(candidate_id="m3", prompt="What is the capital of Peru?", tier=1, task_type="qa_classification", rubric="rules-v2"),
]


def test_only_memory_tier1_labels_need_review() -> None:
    assert [label.candidate_id for label in V2 if needs_fact_review(label)] == ["m1", "m2", "m3"]


def test_round3_queue_puts_test_prompts_first() -> None:
    queue = [c.candidate_id for c in round3_queue(V2, {"m3"})]
    assert queue[0] == "m3" and set(queue) == {"m1", "m2", "m3"}


def test_merge_review_replaces_drops_and_keeps() -> None:
    reviewed = [LabeledPrompt(candidate_id="m1", prompt="Who wrote Hamlet?", tier=2, task_type="qa_classification",
                              rubric=LATEST_RUBRIC)]
    merged = merge_review(V2, reviewed, skipped_ids={"m2"})
    assert [(m.candidate_id, m.tier, m.rubric) for m in merged] == [
        ("m1", 2, "rules-v3"), ("c1", 1, "rules-v2"), ("h1", 3, "rules-v2"), ("m3", 1, "rules-v2"),
    ]


def test_latest_labels_file_prefers_newest(tmp_path: Path) -> None:
    assert latest_labels_file(tmp_path).name == "labeled_prompts.jsonl"  # nothing yet → round 1 path
    (tmp_path / "labeled_prompts_v2.jsonl").write_text("")
    assert latest_labels_file(tmp_path).name == "labeled_prompts_v2.jsonl"
    (tmp_path / "labeled_prompts_v3.jsonl").write_text("")
    assert latest_labels_file(tmp_path).name == "labeled_prompts_v3.jsonl"


def test_round3_script_reviews_merges_and_keeps_v2(tmp_path: Path) -> None:
    write_candidates_csv(tmp_path / "candidates.csv", [Candidate(candidate_id=v.candidate_id, prompt=v.prompt) for v in V2])
    write_jsonl(tmp_path / "labeled_prompts_v2.jsonl", V2)
    (tmp_path / "test_split.json").write_text(
        TestSplit(created_at="2026-01-01T00:00:00Z", seed=1, test_size=0.2, test_ids=["m3"]).model_dump_json()  # type: ignore[arg-type]
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    # m3 (test prompt) first → tier 2; next → skip; then quit.
    keys = Keyboard("2", "2", "", "s", "broken", "q")
    assert load_script("label_prompts").main(["--data-dir", str(tmp_path), "--round", "3"], ask=keys, console=console) == 0

    out = buffer.getvalue()
    assert "Rules (rules-v3)" in out and "answered from memory" in out
    assert "Who is named?" not in out and "Why do we dream?" not in out  # unaffected prompts never shown
    assert "Reviewed 2 of 3" in out and "Agreement with your v2 labels: 0%" in out

    merged = read_jsonl(tmp_path / "labeled_prompts_v3.jsonl", LabeledPrompt)
    by_id = {m.candidate_id: m for m in merged}
    assert by_id["m3"].tier == 2 and by_id["m3"].rubric == "rules-v3"
    assert len(merged) == 4 and by_id["c1"].tier == 1  # one skipped prompt dropped, rest carried over
    assert read_jsonl(tmp_path / "labeled_prompts_v2.jsonl", LabeledPrompt) == V2  # v2 untouched
