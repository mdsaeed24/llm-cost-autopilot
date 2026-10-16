"""Phase 1 smoke test: send the same prompts to every model tier and compare.

Writes data/baseline_comparison.json and prints a comparison table.

    python scripts/smoke_test.py                     # all 10 prompts, all models
    python scripts/smoke_test.py --limit 3           # first 3 prompts only
    python scripts/smoke_test.py --models local-llama deepseek-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from autopilot.evaluation.baseline import BaselineResult, ModelSummary, build_report, run_baseline, summarise
from autopilot.evaluation.smoke_prompts import SMOKE_PROMPTS, SmokePrompt
from autopilot.llm.catalog import load_catalog
from autopilot.llm.client import LLMClient, LLMRequestError
from autopilot.llm.pricing import peak_status
from autopilot.llm.schemas import ChatMessage, ModelConfig
from autopilot.settings import PROJECT_ROOT, load_settings

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only send the first N prompts.")
    parser.add_argument("--models", nargs="+", default=None, help="Model names from config/models.yaml.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Output token cap (default: settings.yaml).")
    parser.add_argument("--concurrency", type=int, default=None, help="Override per-provider concurrency.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "baseline_comparison.json")
    parser.add_argument("--show-outputs", action="store_true", help="Print each model's answer.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip loading the Ollama model before timing.")
    return parser.parse_args(argv)


def worst_case_cost(prompts: list[SmokePrompt], models: list[ModelConfig], max_tokens: int) -> float:
    """Upper bound: every answer runs to max_tokens; input ~ 4 chars per token, all cache misses."""
    total = 0.0
    for model in models:
        for prompt in prompts:
            input_tokens = len(prompt.prompt) / 4 + 20
            total += (
                input_tokens * model.pricing.input_cache_miss_per_1m + max_tokens * model.pricing.output_per_1m
            ) / 1_000_000
    return total


async def warm_up(client: LLMClient, models: list[ModelConfig]) -> None:
    """Load local models into memory so the first timed request isn't a cold start."""
    for model in models:
        if model.provider != "ollama":
            continue
        console.print(f"[dim]Warming up {model.name} ({model.model_id})...[/dim]")
        try:
            await client.send([ChatMessage(role="user", content="Hi")], model, max_tokens=1)
        except LLMRequestError as error:
            console.print(f"[yellow]Warm-up failed: {error}[/yellow]")


def print_summary(summaries: list[ModelSummary]) -> None:
    table = Table(title="Baseline comparison (cost at DeepSeek PEAK prices)")
    for column in ("Model", "Tier", "OK", "Failed", "Truncated", "Total cost", "$ / 1k req",
                   "Mean lat.", "Median lat.", "Max lat.", "Avg out tok", "Contains expected"):
        table.add_column(column, justify="right" if column not in ("Model", "Tier") else "left")
    for s in summaries:
        def ms(value: float | None) -> str:
            return f"{value / 1000:.1f}s" if value is not None else "-"
        table.add_row(
            s.model_name,
            s.quality_tier,
            str(s.n_ok),
            str(s.n_failed),
            str(s.n_truncated),
            f"${s.total_cost_usd:.5f}",
            f"${s.projected_cost_per_1k_usd:.3f}" if s.projected_cost_per_1k_usd is not None else "-",
            ms(s.mean_latency_ms),
            ms(s.median_latency_ms),
            ms(s.max_latency_ms),
            f"{s.mean_output_tokens:.0f}" if s.mean_output_tokens is not None else "-",
            f"{s.expected_hits}/{s.expected_total}",
        )
    console.print(table)


def print_matrix(results: list[BaselineResult], prompts: list[SmokePrompt], models: list[ModelConfig]) -> None:
    table = Table(title="Per prompt: latency / cost  (✓/✗ = contains expected answer, T = truncated)")
    table.add_column("Prompt")
    table.add_column("Kind")
    for model in models:
        table.add_column(model.name, justify="right")
    by_key = {(r.prompt_id, r.model_name): r for r in results}
    for prompt in prompts:
        cells = []
        for model in models:
            result = by_key[(prompt.prompt_id, model.name)]
            if result.response is None:
                cells.append("[red]error[/red]")
                continue
            r = result.response
            marks = {True: " ✓", False: " [red]✗[/red]", None: ""}[result.contains_expected]
            trunc = " [yellow]T[/yellow]" if r.truncated else ""
            cells.append(f"{r.latency_ms / 1000:.1f}s / ${r.cost_usd:.5f}{marks}{trunc}")
        table.add_row(prompt.prompt_id, prompt.kind, *cells)
    console.print(table)


def print_outputs(results: list[BaselineResult]) -> None:
    for result in results:
        console.rule(f"{result.prompt_id} · {result.model_name}")
        if result.response is None:
            console.print(f"[red]{result.error}[/red]")
        else:
            console.print(result.response.output_text.strip(), markup=False)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    catalog = load_catalog()
    settings = load_settings().client
    if args.concurrency:
        settings = settings.model_copy(update={"concurrency_default": args.concurrency, "concurrency_per_provider": {}})
    max_tokens = args.max_tokens or settings.max_output_tokens

    models = [catalog.get(name) for name in args.models] if args.models else catalog.by_quality()
    skipped = {m.name: f"{m.api_key_env} not set" for m in models if not m.has_api_key()}
    for name, reason in skipped.items():
        console.print(f"[yellow]Skipping {name}: {reason} (see .env.example)[/yellow]")
    models = [m for m in models if m.name not in skipped]
    if not models:
        console.print("[red]No callable models.[/red]")
        return 1

    prompts = list(SMOKE_PROMPTS[: args.limit] if args.limit else SMOKE_PROMPTS)
    status = peak_status(catalog.price_schedule)
    console.print(status.describe())
    console.print(
        f"Sending {len(prompts)} prompts × {len(models)} models (max_tokens={max_tokens}). "
        f"Worst-case cost: ${worst_case_cost(prompts, models, max_tokens):.4f}"
    )

    client = LLMClient(settings)
    try:
        if not args.no_warmup:
            await warm_up(client, models)
        results = await run_baseline(client, prompts, models, max_tokens=max_tokens)
    finally:
        await client.aclose()

    summaries = summarise(results, models)
    report = build_report(
        results, summaries, prompts, skipped_models=skipped, pricing_status=status.describe(), max_tokens=max_tokens
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.show_outputs:
        print_outputs(results)
    print_matrix(results, prompts, models)
    print_summary(summaries)
    console.print(f"Saved → {args.output.relative_to(PROJECT_ROOT) if args.output.is_relative_to(PROJECT_ROOT) else args.output}")
    return 0 if all(r.response is not None for r in results) else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
