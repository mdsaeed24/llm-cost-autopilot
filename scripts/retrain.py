"""Feedback loop: retrain on human labels + confirmed routing failures; promote only if not worse.

Each routing failure becomes a training example labeled one tier above the tier that failed.
The candidate and the current model are scored on the SAME frozen test prompts. Every run is
appended to models/retrain_log.jsonl.

    python scripts/retrain.py                       # retrain if there is at least 1 new failure
    python scripts/retrain.py --dry-run             # train + compare, save nothing
    python scripts/retrain.py --min-new-failures 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from autopilot.classifier.feedback import failures_to_labels
from autopilot.classifier.features import FEATURE_SETS
from autopilot.classifier.registry import DEFAULT_MODELS_DIR
from autopilot.classifier.reporting import print_outcome, print_report
from autopilot.classifier.training import DEFAULT_SPECS, TestSplit
from autopilot.classifier.workflow import RetrainLogEntry, train_compare_promote
from autopilot.dataset.io import read_jsonl, write_jsonl
from autopilot.dataset.schemas import LabeledPrompt
from autopilot.settings import PROJECT_ROOT, load_settings
from autopilot.storage.db import Database, utc_now
from autopilot.storage.repository import Repository


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = PROJECT_ROOT / "data"
    parser.add_argument("--labels", type=Path, nargs="+", default=[data / "labeled_prompts_v2.jsonl"])
    parser.add_argument("--split", type=Path, default=data / "test_split.json")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--failures-out", type=Path, default=data / "routing_failures.jsonl")
    parser.add_argument("--db", type=Path, default=None, help="SQLite file (default: settings.yaml).")
    parser.add_argument("--min-new-failures", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.0, help="Allowed drop when comparing models.")
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _summary(metrics: object) -> dict[str, float]:
    return {
        "accuracy": round(metrics.accuracy, 4),  # type: ignore[attr-defined]
        "tier3_recall": round(metrics.tier3_recall, 4),  # type: ignore[attr-defined]
        "under_routing_rate": round(metrics.under_routing_rate, 4),  # type: ignore[attr-defined]
    }


def main(argv: list[str] | None = None, *, console: Console | None = None) -> int:
    args = parse_args(argv)
    console = console or Console()
    logging.basicConfig(level=logging.WARNING)

    repo = Repository(Database(args.db or load_settings().database.resolved_path()))
    human = [label for path in args.labels for label in read_jsonl(path, LabeledPrompt)]
    test_ids = set(TestSplit.model_validate_json(args.split.read_text()).test_ids)
    test_prompts = [h.prompt for h in human if h.candidate_id in test_ids]

    failures = repo.list_routing_failures()
    feedback = failures_to_labels(failures, test_prompts=test_prompts, human_prompts=[h.prompt for h in human])
    usable = set(feedback.failure_ids)
    new = [f for f in failures if f.id in usable and f.used_in_version is None]
    console.print(
        f"Routing failures in DB: {len(failures)} → usable examples: {len(feedback.examples)} "
        f"({len(new)} new) · dropped: test-set overlap {feedback.dropped_test_overlap}, "
        f"already human-labeled {feedback.dropped_human_overlap}, duplicates {feedback.dropped_duplicate}, "
        f"no task type {feedback.dropped_no_task_type}"
    )
    if len(new) < args.min_new_failures:
        console.print(f"[yellow]Only {len(new)} new failures (< {args.min_new_failures}); nothing to retrain.[/yellow]")
        return 0

    write_jsonl(args.failures_out, feedback.examples)
    outcome = train_compare_promote(
        [*args.labels, args.failures_out],
        split=args.split,
        models_dir=args.models_dir,
        specs=[s for s in DEFAULT_SPECS if s.features in args.feature_sets],
        folds=args.folds,
        save=not args.dry_run,
        tolerance=args.tolerance,
    )
    print_report(outcome.trained.report, console, split_name=args.split.name)
    print_outcome(outcome, console, models_dir_name=args.models_dir.name)

    entry = RetrainLogEntry(
        at=utc_now(),
        dry_run=args.dry_run,
        candidate_version=outcome.metadata.version if outcome.metadata else None,
        current_version=outcome.current_version,
        human_examples=len(human),
        failure_examples=len(feedback.examples),
        new_failures=len(new),
        candidate=_summary(outcome.decision.candidate),
        current=_summary(outcome.decision.current) if outcome.decision.current else None,
        decision=outcome.promotion_reason,
        promoted=outcome.promoted,
    )
    args.models_dir.mkdir(parents=True, exist_ok=True)
    with (args.models_dir / "retrain_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(entry.model_dump_json() + "\n")
    if outcome.metadata is not None:
        repo.mark_failures_used(feedback.failure_ids, outcome.metadata.version)
    console.print(f"Logged → {args.models_dir.name}/retrain_log.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
