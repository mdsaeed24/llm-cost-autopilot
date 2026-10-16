"""Send prompts through the autopilot — routing, answering, logging, verification — without the API.

    python scripts/ask.py "Why is the sky blue?"
    python scripts/ask.py --loadtest 10 --mode evaluation         # 10 pool prompts, verify all
    python scripts/ask.py --loadtest 10 --escalation sync --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from autopilot.bootstrap import build_service
from autopilot.llm.catalog import load_catalog
from autopilot.llm.pricing import peak_status
from autopilot.service import AutopilotService, CompletionResult
from autopilot.settings import PROJECT_ROOT, AppSettings, load_settings

console = Console()

# Rough per-request token sizes for the up-front estimate.
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS, EST_JUDGE_INPUT, EST_JUDGE_OUTPUT = 400, 450, 1400, 120


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", nargs="*", help="Prompts to send.")
    parser.add_argument("--loadtest", type=int, default=0, metavar="N", help="Use the first N load-test prompts.")
    parser.add_argument("--mode", choices=("normal", "evaluation"), default=None)
    parser.add_argument("--escalation", choices=("async", "sync"), default=None)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=3, help="Requests in flight at once.")
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation.")
    return parser.parse_args(argv)


def apply_overrides(settings: AppSettings, args: argparse.Namespace) -> AppSettings:
    update: dict[str, object] = {}
    if args.mode:
        update["mode"] = args.mode
    if args.escalation:
        update["escalation_mode"] = args.escalation
    if args.sample_rate is not None:
        update["sample_rate_override"] = args.sample_rate
    return settings.model_copy(update={"verification": settings.verification.model_copy(update=update)})


def estimate_cost(n: int, settings: AppSettings) -> float:
    """Upper-ish estimate: every request on Tier 3, plus verification at the sample rate."""
    pro = load_catalog().get("deepseek-v4-pro").pricing
    answer = (EST_INPUT_TOKENS * pro.input_cache_miss_per_1m + EST_OUTPUT_TOKENS * pro.output_per_1m) / 1e6
    judge = (EST_JUDGE_INPUT * pro.input_cache_miss_per_1m + EST_JUDGE_OUTPUT * pro.output_per_1m) / 1e6
    return n * (answer + settings.verification.sample_rate * (answer + judge))


async def run_all(service: AutopilotService, prompts: Sequence[str], concurrency: int) -> list[CompletionResult | Exception]:
    gate = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> CompletionResult | Exception:
        async with gate:
            try:
                return await service.complete([{"role": "user", "content": prompt}])
            except Exception as error:  # noqa: BLE001 - report per prompt, keep going
                return error

    return list(await asyncio.gather(*(one(p) for p in prompts)))


def print_results(prompts: Sequence[str], results: Sequence[CompletionResult | Exception], show: bool) -> None:
    table = Table(title="Requests")
    for col in ("#", "prompt", "pred", "conf", "routed → served", "cost", "latency", "verification", "esc"):
        table.add_column(col, overflow="fold" if col == "prompt" else "ellipsis")
    for i, (prompt, result) in enumerate(zip(prompts, results, strict=True), start=1):
        preview = " ".join(prompt.split())[:60]
        if isinstance(result, Exception):
            table.add_row(str(i), preview, "-", "-", f"[red]{type(result).__name__}[/red]", "-", "-", "-", "-")
            continue
        d = result.decision
        up = "↑" if d.routed_up else ""
        route = d.model_name if result.served_model == d.model_name else f"{d.model_name} → {result.served_model}"
        table.add_row(
            str(i), preview, str(d.predicted_tier), f"{d.confidence:.2f}" if d.confidence is not None else "-",
            f"{route}{up}", f"${result.response.cost_usd:.5f}", f"{result.latency_ms / 1000:.1f}s",
            result.verification_status, "yes" if result.escalated else "",
        )
    console.print(table)
    if show:
        for i, result in enumerate(results, start=1):
            if not isinstance(result, Exception):
                console.rule(f"#{i} · {result.served_model}")
                console.print(result.output_text.strip(), markup=False)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    settings = apply_overrides(load_settings(), args)

    prompts = list(args.prompts)
    if args.loadtest:
        with (PROJECT_ROOT / "data" / "loadtest_prompts.jsonl").open(encoding="utf-8") as handle:
            prompts += [json.loads(line)["prompt"] for _, line in zip(range(args.loadtest), handle, strict=False)]
    if not prompts:
        console.print("[red]Give prompts or --loadtest N[/red]")
        return 1

    v = settings.verification
    console.print(peak_status(load_catalog().price_schedule).describe())
    console.print(
        f"{len(prompts)} prompts · mode={v.mode} · sample rate={v.sample_rate:.0%} · escalation={v.escalation_mode} · "
        f"estimated cost ≤ ${estimate_cost(len(prompts), settings):.3f} (+ background verification by the worker)"
    )
    if not args.yes and estimate_cost(len(prompts), settings) > 0.05:
        if input("Continue? [y/N] ").strip().lower() != "y":
            return 1

    service = build_service(settings)
    try:
        results = await run_all(service, prompts, args.concurrency)
    finally:
        await service.client.aclose()
    print_results(prompts, results, args.show_answers)
    console.print(f"DB summary: {service.repo.verification_summary()}")
    console.print(f"Queue: {service.repo.queue_counts()} — run `make worker-once` to verify queued requests.")
    return 0 if not any(isinstance(r, Exception) for r in results) else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
