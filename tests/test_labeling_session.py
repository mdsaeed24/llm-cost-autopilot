"""The interactive labeling loop, driven by scripted keyboard input."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from autopilot.dataset.io import write_candidates_csv
from autopilot.dataset.labeling import run_session
from autopilot.dataset.labels import LabelStore
from autopilot.dataset.schemas import Candidate
from tests.conftest import load_script

CANDIDATES = [Candidate(candidate_id=f"c{i:04d}", prompt=f"Prompt number {i} [not markup]") for i in range(1, 6)]


class Keyboard:
    """Feeds scripted answers; raises EOFError when they run out (like closing the terminal)."""

    def __init__(self, *answers: str) -> None:
        self._answers: Iterator[str] = iter(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            return next(self._answers)
        except StopIteration:
            raise EOFError from None


def quiet_console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=120, force_terminal=False), buffer


@pytest.fixture
def store(tmp_path: Path) -> LabelStore:
    return LabelStore(tmp_path / "labeled.jsonl", tmp_path / "skipped.jsonl")


def test_label_skip_and_quit(store: LabelStore) -> None:
    console, _ = quiet_console()
    keys = Keyboard(
        "3", "4", "needs multi-step reasoning",   # c0001 → tier 3, reasoning_creative, note
        "s", "no context given",                  # c0002 → skip
        "1", "extr", "",                          # c0003 → tier 1, task by name prefix, no note
        "q",
    )
    result = run_session(store, CANDIDATES, ask=keys, console=console)

    assert (result.labeled, result.skipped) == (2, 1)
    first, second = store.labels
    assert (first.candidate_id, first.tier, first.task_type, first.note) == ("c0001", 3, "reasoning_creative", "needs multi-step reasoning")
    assert (second.candidate_id, second.tier, second.task_type, second.note) == ("c0003", 1, "extraction_reformatting", None)
    assert store.skips[0].note == "no context given"


def test_invalid_input_is_reprompted_and_back_works(store: LabelStore) -> None:
    console, output = quiet_console()
    keys = Keyboard("7", "2", "9", "b", "2", "3", "", "q")
    run_session(store, CANDIDATES, ask=keys, console=console)
    assert store.labels[0].tier == 2
    assert store.labels[0].task_type == "summarization_analysis"
    assert "Type 1, 2, 3" in output.getvalue()
    assert "Type 1-4" in output.getvalue()


def test_undo_shows_candidate_again(store: LabelStore) -> None:
    console, _ = quiet_console()
    keys = Keyboard("1", "1", "", "u", "2", "2", "", "q")  # label c0001 as 1, undo, relabel as 2
    result = run_session(store, CANDIDATES, ask=keys, console=console)
    assert result.undone == 1
    assert [(r.candidate_id, r.tier) for r in store.labels] == [("c0001", 2)]


def test_resume_skips_done_candidates(store: LabelStore, tmp_path: Path) -> None:
    console, _ = quiet_console()
    run_session(store, CANDIDATES, ask=Keyboard("1", "1", "", "s", ""), console=console)  # then EOF

    resumed = LabelStore(tmp_path / "labeled.jsonl", tmp_path / "skipped.jsonl")
    keys = Keyboard("2", "2", "", "q")
    run_session(resumed, CANDIDATES, ask=keys, console=console)
    assert [r.candidate_id for r in resumed.labels] == ["c0001", "c0003"]


def test_limit_and_finish(store: LabelStore) -> None:
    console, _ = quiet_console()
    result = run_session(store, CANDIDATES, ask=Keyboard(*["1", "1", ""] * 2), console=console, limit=2)
    assert result.labeled == 2 and not result.finished_all

    result = run_session(store, CANDIDATES, ask=Keyboard(*["s", ""] * 3), console=console)
    assert result.finished_all
    assert len(store.done_ids) == 5


def test_ctrl_c_keeps_saved_work(store: LabelStore) -> None:
    console, output = quiet_console()

    def ask(prompt: str) -> str:
        if store.labels:
            raise KeyboardInterrupt
        return {"Tier": "3", "Task": "4", "Note": ""}[prompt.split()[0]]

    run_session(store, CANDIDATES, ask=ask, console=console)
    assert len(store.labels) == 1
    assert "saved" in output.getvalue()


def test_prompt_text_is_not_rendered_as_markup(store: LabelStore) -> None:
    console, output = quiet_console()
    run_session(store, CANDIDATES, ask=Keyboard("q"), console=console)
    assert "[not markup]" in output.getvalue()


def test_label_script_stats_and_session(tmp_path: Path) -> None:
    script = load_script("label_prompts")
    write_candidates_csv(tmp_path / "candidates.csv", CANDIDATES)
    console, output = quiet_console()

    assert script.main(["--data-dir", str(tmp_path)], ask=Keyboard("3", "4", "", "q"), console=console) == 0
    assert (tmp_path / "labeled_prompts.jsonl").exists()
    assert "Tiers" in output.getvalue()  # help shown on first run

    console, output = quiet_console()
    assert script.main(["--data-dir", str(tmp_path), "--stats"], console=console) == 0
    assert "1 labeled" in output.getvalue()
    assert "aim for 40+" in output.getvalue()


def test_label_script_without_candidates(tmp_path: Path) -> None:
    console, output = quiet_console()
    assert load_script("label_prompts").main(["--data-dir", str(tmp_path)], console=console) == 1
    assert "sample_prompts.py" in output.getvalue()
