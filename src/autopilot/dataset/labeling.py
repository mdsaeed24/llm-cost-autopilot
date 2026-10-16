"""Interactive, resumable labeling loop (terminal UI). I/O is injected so it can be tested."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autopilot.dataset.labels import LabelStore
from autopilot.dataset.schemas import (
    TASK_TYPE_HELP,
    TASK_TYPES,
    TIER_HELP,
    TIERS,
    Candidate,
    LabeledPrompt,
    SkippedCandidate,
    TaskType,
    Tier,
    rubric_text,
)

Ask = Callable[[str], str]

TIER3_TARGET = 40
PER_TIER_MINIMUM = 30

GUIDELINES = (
    "• Label what the task REQUIRES, not how long it is. A long text to reformat can still be Tier 1.\n"
    "• Ask: what does a good answer need? Copying/locating facts → 1; condensing or organising → 2;\n"
    "  multi-step reasoning, originality, or weighing trade-offs → 3.\n"
    "• Write a note on borderline cases — they are gold for error analysis later.\n"
    "• Skip (s) prompts that are broken, unanswerable, or need missing context."
)

RUBRIC_GUIDELINES = (
    "• Apply the rules above in order; the first rule that matches gives the tier.\n"
    "• Judge the prompt's wording only — not how obscure the topic is or how long the text is.\n"
    "• Write a note when no rule fits well — those prompts show where the rules need work.\n"
    "• Skip (s) prompts that are broken, unanswerable, or need missing context."
)


class _Action(Enum):
    LABELED = "labeled"
    SKIPPED = "skipped"
    UNDONE = "undone"
    QUIT = "quit"


@dataclass
class SessionResult:
    """What happened during one labeling session."""

    labeled: int = 0
    skipped: int = 0
    undone: int = 0
    finished_all: bool = False


def print_help(console: Console, *, rubric: str | None = None) -> None:
    """Show tier guidance. With a rubric, the rules replace the looser round-1 tier text and
    guidelines, so the labeler never sees two definitions that disagree."""
    if rubric:
        console.print(Panel(rubric_text(), title=f"Rules ({rubric}) — first match decides", expand=False))
    else:
        table = Table(title="Tiers", show_header=False, box=None)
        for tier in TIERS:
            table.add_row(f"[bold]{tier}[/bold]", TIER_HELP[tier])
        console.print(table)
    tasks = Table(title="Task types", show_header=False, box=None)
    for number, task in enumerate(TASK_TYPES, start=1):
        tasks.add_row(f"[bold]{number}[/bold]", task, f"[dim]{TASK_TYPE_HELP[task]}[/dim]")
    console.print(tasks)
    console.print(Panel(RUBRIC_GUIDELINES if rubric else GUIDELINES, title="Guidelines", expand=False))
    console.print("[dim]Keys: 1/2/3 tier · s skip · u undo last · q quit (progress is saved) · ? help[/dim]")


def print_stats(store: LabelStore, total: int, console: Console) -> None:
    """Label counts so far. Only shown at the end of a session, never while labeling,
    so that "I need more Tier 3s" can't creep into individual judgments."""
    tiers = store.tier_counts()
    console.print(f"[bold]Progress: {len(store.labels)} labeled · {len(store.skips)} skipped · {total} candidates[/bold]")
    table = Table()
    table.add_column("Tier")
    table.add_column("Count", justify="right")
    for tier in TIERS:
        table.add_row(str(tier), str(tiers.get(tier, 0)))
    console.print(table)
    tasks = store.task_type_counts()
    console.print("Task types: " + ", ".join(f"{t}={tasks.get(t, 0)}" for t in TASK_TYPES))
    if tiers.get(3, 0) < TIER3_TARGET:
        console.print(f"[yellow]Tier 3 has {tiers.get(3, 0)} examples; aim for {TIER3_TARGET}+ (Tier-3 recall matters most).[/yellow]")
    thin = [str(t) for t in TIERS if tiers.get(t, 0) < PER_TIER_MINIMUM]
    if thin and store.labels:
        console.print(f"[yellow]Tiers with fewer than {PER_TIER_MINIMUM} examples: {', '.join(thin)}[/yellow]")


def _parse_task_type(raw: str) -> TaskType | None:
    raw = raw.strip().lower()
    if raw.isdigit() and 1 <= int(raw) <= len(TASK_TYPES):
        return TASK_TYPES[int(raw) - 1]
    matches = [t for t in TASK_TYPES if t.startswith(raw)] if raw else []
    return matches[0] if len(matches) == 1 else None


def _show_candidate(
    candidate: Candidate, position: int, total: int, console: Console, *, rubric: str | None
) -> None:
    console.print()
    if rubric:
        console.print(Text(rubric_text(), style="dim"))
    # Text(...) so brackets in the prompt are never interpreted as rich markup.
    console.print(Panel(Text(candidate.prompt), title=f"{candidate.candidate_id}  ·  {position}/{total}", title_align="left"))


def _label_one(
    candidate: Candidate, store: LabelStore, ask: Ask, console: Console, *, rubric: str | None = None
) -> _Action:
    """Ask for tier → task type → note. Returns what the user did."""
    while True:
        choice = ask("Tier [1/2/3] · s skip · u undo · q quit · ? help > ").strip().lower()
        if choice == "q":
            return _Action.QUIT
        if choice == "?":
            print_help(console, rubric=rubric)
            continue
        if choice == "u":
            undone = store.undo_last()
            if undone is None:
                console.print("[dim]Nothing to undo.[/dim]")
                continue
            console.print(f"[cyan]↺ undid {undone} — it will be shown again[/cyan]")
            return _Action.UNDONE
        if choice == "s":
            note = ask("Why skip? (optional) > ").strip() or None
            store.add_skip(SkippedCandidate(candidate_id=candidate.candidate_id, note=note))
            console.print(f"[dim]⤼ skipped {candidate.candidate_id}[/dim]")
            return _Action.SKIPPED
        if choice in ("1", "2", "3"):
            tier: Tier = int(choice)  # type: ignore[assignment]
            task_type = _ask_task_type(ask, console)
            if task_type is None:
                continue  # 'b' → back to tier question
            note = ask("Note (optional, Enter to skip) > ").strip() or None
            store.add_label(
                LabeledPrompt(
                    candidate_id=candidate.candidate_id,
                    prompt=candidate.prompt,
                    tier=tier,
                    task_type=task_type,
                    note=note,
                    rubric=rubric,
                )
            )
            console.print(f"[green]✓ {candidate.candidate_id} → Tier {tier} · {task_type}[/green]")
            return _Action.LABELED
        console.print("[red]Type 1, 2, 3, s, u, q or ?[/red]")


def _ask_task_type(ask: Ask, console: Console) -> TaskType | None:
    menu = "  ".join(f"{i}={t}" for i, t in enumerate(TASK_TYPES, start=1))
    while True:
        raw = ask(f"Task type [{menu}] · b back > ")
        if raw.strip().lower() == "b":
            return None
        task_type = _parse_task_type(raw)
        if task_type is not None:
            return task_type
        console.print("[red]Type 1-4 (or the start of a task-type name), or b to go back[/red]")


def run_session(
    store: LabelStore,
    candidates: Sequence[Candidate],
    *,
    ask: Ask,
    console: Console,
    limit: int | None = None,
    rubric: str | None = None,
) -> SessionResult:
    """Show unlabeled candidates in file order until done, quit, or ``limit`` labels/skips.

    Every answer is saved before the next prompt is shown, so quitting (q, Ctrl+C,
    closing the terminal) never loses work. Re-running resumes where you stopped.
    """
    result = SessionResult()
    total = len(candidates)
    try:
        while limit is None or result.labeled + result.skipped < limit:
            done = store.done_ids
            pending = next((c for c in candidates if c.candidate_id not in done), None)
            if pending is None:
                result.finished_all = True
                console.print("[bold green]All candidates done![/bold green]")
                break
            _show_candidate(pending, len(done) + 1, total, console, rubric=rubric)
            action = _label_one(pending, store, ask, console, rubric=rubric)
            if action is _Action.QUIT:
                break
            if action is _Action.LABELED:
                result.labeled += 1
            elif action is _Action.SKIPPED:
                result.skipped += 1
            elif action is _Action.UNDONE:
                result.undone += 1
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Interrupted — everything up to the last answer is saved.[/dim]")
    return result
