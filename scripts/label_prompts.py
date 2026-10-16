"""Label candidates one at a time: tier (1/2/3), task type, optional note.

Saves after every answer; stop any time with q or Ctrl+C and re-run to resume.

Round 1 (original):  data/labeled_prompts.jsonl
Round 2 (--round 2): re-label with the written rules → data/labeled_prompts_v2.jsonl.
                     Round-1 labels are never shown and never modified.

    python scripts/label_prompts.py                    # round 1
    python scripts/label_prompts.py --round 2          # round 2 with the rules on screen
    python scripts/label_prompts.py --round 2 --stats  # progress + agreement with round 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from autopilot.classifier.training import TestSplit
from autopilot.dataset.io import read_candidates_csv, read_jsonl
from autopilot.dataset.labeling import Ask, print_help, print_stats, run_session
from autopilot.dataset.labels import LabelStore
from autopilot.dataset.relabel import agreement, round2_queue
from autopilot.dataset.schemas import RUBRIC_VERSION, TIERS, LabeledPrompt
from autopilot.settings import PROJECT_ROOT


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N labels/skips this session.")
    parser.add_argument("--stats", action="store_true", help="Show progress and exit.")
    return parser.parse_args(argv)


def print_agreement(previous: list[LabeledPrompt], current: list[LabeledPrompt], console: Console) -> None:
    result = agreement(previous, current)
    if result.n == 0:
        return
    console.print(
        f"\n[bold]Agreement with your round-1 labels: {result.rate:.0%}[/bold] ({result.same}/{result.n}) · "
        f"moved up {result.moved_up} · moved down {result.moved_down}"
    )
    table = Table(title="Round 1 (rows) vs round 2 (columns)")
    table.add_column("R1 ↓ / R2 →")
    for tier in TIERS:
        table.add_column(str(tier), justify="right")
    for tier, row in zip(TIERS, result.confusion, strict=True):
        table.add_row(str(tier), *(str(v) for v in row))
    console.print(table)


def main(argv: list[str] | None = None, *, ask: Ask | None = None, console: Console | None = None) -> int:
    args = parse_args(argv)
    console = console or Console()
    data_dir: Path = args.data_dir
    candidates_path = data_dir / "candidates.csv"
    if not candidates_path.exists():
        console.print(f"[red]{candidates_path} not found — run scripts/sample_prompts.py first.[/red]")
        return 1
    candidates = read_candidates_csv(candidates_path)

    previous: list[LabeledPrompt] = []
    rubric: str | None = None
    if args.round == 1:
        store = LabelStore(data_dir / "labeled_prompts.jsonl", data_dir / "skipped_candidates.jsonl")
    else:
        previous = read_jsonl(data_dir / "labeled_prompts.jsonl", LabeledPrompt)
        split_path = data_dir / "test_split.json"
        test_ids = TestSplit.model_validate_json(split_path.read_text()).test_ids if split_path.exists() else []
        candidates = round2_queue(candidates, previous, set(test_ids))
        store = LabelStore(data_dir / "labeled_prompts_v2.jsonl", data_dir / "skipped_candidates_v2.jsonl")
        rubric = RUBRIC_VERSION

    if not args.stats:
        if not store.done_ids:
            print_help(console, rubric=rubric)
        ask = ask or (lambda prompt: console.input(Text(prompt, style="bold")))
        result = run_session(store, candidates, ask=ask, console=console, limit=args.limit, rubric=rubric)
        console.print(f"\nThis session: {result.labeled} labeled, {result.skipped} skipped, {result.undone} undone.")
    print_stats(store, len(candidates), console)
    if args.round == 2:
        print_agreement(previous, store.labels, console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
