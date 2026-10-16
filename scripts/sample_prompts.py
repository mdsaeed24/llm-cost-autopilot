"""Sample labeling candidates and a load-test pool from Databricks Dolly 15k (CC BY-SA 3.0).

Writes:
  data/candidates.csv            candidate_id, prompt   (stratified, ~40% hard-prompt strata)
  data/loadtest_prompts.jsonl    1,000 prompts, uniform random, no overlap with candidates
  data/raw/sample_manifest.json  provenance (seed, params, Dolly row indices) — not for labeling

    python scripts/sample_prompts.py
    python scripts/sample_prompts.py --n-candidates 150 --loadtest-size 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from autopilot.dataset.io import atomic_write_text, write_candidates_csv, write_jsonl
from autopilot.dataset.sampling import (
    DEFAULT_QUOTAS,
    DOLLY_DATASET,
    DOLLY_LICENSE,
    TIER3_LIKELY_STRATA,
    DollyRecord,
    clean_records,
    load_dolly,
    sample_pool,
    scale_quotas,
    stratified_sample,
    to_candidates,
    to_loadtest,
)
from autopilot.settings import PROJECT_ROOT

console = Console()
Loader = Callable[[Path], list[DollyRecord]]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-candidates", type=int, default=sum(DEFAULT_QUOTAS.values()))
    parser.add_argument("--loadtest-size", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=12, help="Drop junk prompts shorter than this.")
    parser.add_argument(
        "--max-chars", type=int, default=6000,
        help="Drop prompts longer than this (~1,500 tokens): keeps labeling sane and stays inside "
             "Ollama's default context window.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite candidates even if labels exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, loader: Loader = load_dolly) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir
    candidates_path = data_dir / "candidates.csv"
    loadtest_path = data_dir / "loadtest_prompts.jsonl"
    labels_path = data_dir / "labeled_prompts.jsonl"
    manifest_path = data_dir / "raw" / "sample_manifest.json"

    if candidates_path.exists() and labels_path.exists() and labels_path.stat().st_size > 0 and not args.force:
        console.print(
            f"[red]{labels_path.name} already has labels. Re-sampling would give candidate ids new meanings.[/red]\n"
            "Re-run with --force only if you intend to start labeling over."
        )
        return 1

    console.print(f"Loading {DOLLY_DATASET} (cached in data/raw/)...")
    records = loader(data_dir / "raw" / "hf_cache")
    cleaned = clean_records(records, min_chars=args.min_chars, max_chars=args.max_chars)
    console.print(f"{len(records)} rows → {len(cleaned)} after length filter + de-duplication")

    rng = random.Random(args.seed)
    quotas = scale_quotas(DEFAULT_QUOTAS, args.n_candidates)
    sample, shortfall = stratified_sample(cleaned, quotas, rng)
    pool = sample_pool(cleaned, exclude={r.index for r in sample}, size=args.loadtest_size, rng=rng)

    candidates = to_candidates(sample)
    loadtest = to_loadtest(pool)
    write_candidates_csv(candidates_path, candidates)
    write_jsonl(loadtest_path, loadtest)

    strata = Counter(r.stratum for r in sample)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": DOLLY_DATASET,
        "license": DOLLY_LICENSE,
        "seed": args.seed,
        "params": {k: v for k, v in vars(args).items() if k not in ("data_dir", "force")},
        "rows_total": len(records),
        "rows_after_cleaning": len(cleaned),
        "quotas": quotas,
        "shortfall": shortfall,
        "candidate_strata_counts": dict(sorted(strata.items())),
        "candidate_dolly_index": {c.candidate_id: r.index for c, r in zip(candidates, sample, strict=True)},
        "loadtest_dolly_index": {p.loadtest_id: r.index for p, r in zip(loadtest, pool, strict=True)},
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

    hard = sum(strata[s] for s in TIER3_LIKELY_STRATA)
    console.print(f"[green]✓ {len(candidates)} candidates → {candidates_path.relative_to(data_dir.parent)}[/green]")
    console.print(f"  {hard} ({hard / len(candidates):.0%}) from Tier-3-likely strata (creative, brainstorming, open reasoning)")
    console.print(f"[green]✓ {len(loadtest)} load-test prompts → {loadtest_path.relative_to(data_dir.parent)}[/green] (uniform random, no overlap)")
    if shortfall:
        console.print(f"[yellow]Strata with too few rows: {shortfall}[/yellow]")
    console.print("[dim]Dolly categories are recorded only in data/raw/ and are never shown during labeling.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
