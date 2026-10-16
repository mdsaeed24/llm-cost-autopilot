"""Train the complexity classifier on hand-labeled prompts, compare models, save a new version.

The new version becomes current only if it is not worse than the current model on the same
frozen test prompts (Tier-3 recall first, then accuracy).

    python scripts/train_classifier.py --labels data/labeled_prompts_v2.jsonl
    python scripts/train_classifier.py --feature-sets tfidf_hc      # skip embeddings (no Ollama needed)
    python scripts/train_classifier.py --promote                    # promote regardless of the comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from autopilot.classifier.features import FEATURE_SETS
from autopilot.classifier.registry import DEFAULT_MODELS_DIR
from autopilot.classifier.reporting import print_outcome, print_report
from autopilot.classifier.training import DEFAULT_SPECS
from autopilot.classifier.workflow import train_compare_promote
from autopilot.dataset.relabel import latest_labels_file
from autopilot.settings import PROJECT_ROOT

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = PROJECT_ROOT / "data"
    parser.add_argument("--labels", type=Path, nargs="+", default=[latest_labels_file(data)],
                        help="Labels file(s); default: the newest (v3 → v2 → round 1)")
    parser.add_argument("--split", type=Path, default=data / "test_split.json", help="Frozen test-set ids.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-accuracy", type=float, default=0.70, help="CV accuracy floor for model selection.")
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS),
                        help="Restrict which feature sets are compared.")
    promotion = parser.add_mutually_exclusive_group()
    promotion.add_argument("--promote", action="store_true", help="Promote regardless of the comparison.")
    promotion.add_argument("--no-promote", action="store_true", help="Save only; never promote.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, console: Console | None = None) -> int:
    args = parse_args(argv)
    console = console or Console()
    logging.basicConfig(level=logging.WARNING)

    console.print(f"Training on {', '.join(p.name for p in args.labels)}")
    outcome = train_compare_promote(
        args.labels,
        split=args.split,
        models_dir=args.models_dir,
        specs=[s for s in DEFAULT_SPECS if s.features in args.feature_sets],
        folds=args.folds,
        seed=args.seed,
        test_size=args.test_size,
        min_accuracy=args.min_accuracy,
        promote_mode="force" if args.promote else "never" if args.no_promote else "auto",
    )
    print_report(outcome.trained.report, console, split_name=args.split.name)
    print_outcome(outcome, console, models_dir_name=args.models_dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
