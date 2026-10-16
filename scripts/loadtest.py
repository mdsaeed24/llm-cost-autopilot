"""Load test: send N prompts from data/loadtest_prompts.jsonl to the running API in evaluation mode.

Prerequisite: the stack runs in evaluation mode (every downgraded answer gets verified):
    make up-eval

    python scripts/loadtest.py --limit 50          # small run first
    python scripts/loadtest.py --limit 1000        # full run (asks before spending)

Writes data/loadtest_report.json and data/loadtest_report.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from autopilot.classifier.registry import load_classifier
from autopilot.dataset.io import read_jsonl
from autopilot.dataset.schemas import LoadtestPrompt
from autopilot.evaluation.loadtest import (
    LoadtestAbort,
    build_report,
    check_api,
    estimate_cost,
    fetch_stats,
    now_utc,
    render_markdown,
    send_all,
    wait_for_verification,
)
from autopilot.llm.catalog import load_catalog
from autopilot.llm.pricing import peak_status
from autopilot.routing.config import load_routing_config
from autopilot.routing.router import Router
from autopilot.settings import PROJECT_ROOT, load_settings

console = Console()
DATA = PROJECT_ROOT / "data"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=1000, help="Number of prompts to send (max 1000).")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N pool prompts.")
    parser.add_argument("--concurrency", type=int, default=5, help="Requests in flight at once.")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("AUTOPILOT_API_KEY") or None)
    parser.add_argument("--wait-timeout-min", type=float, default=120, help="Max wait for verification to finish.")
    parser.add_argument("--prompts", type=Path, default=DATA / "loadtest_prompts.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "loadtest_report.json")
    parser.add_argument("--md", type=Path, default=DATA / "loadtest_report.md")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args(argv)


def confirm(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() == "y"


async def main(argv: list[str] | None = None, *, transport: httpx.AsyncBaseTransport | None = None) -> int:
    args = parse_args(argv)
    catalog = load_catalog()
    prompts = read_jsonl(args.prompts, LoadtestPrompt)[args.offset : args.offset + args.limit]
    if not prompts:
        console.print("[red]No prompts selected.[/red]")
        return 1

    status = peak_status(catalog.price_schedule)
    console.print(status.describe())
    if status.is_peak:
        console.print(
            f"[yellow]⚠ You are running during DeepSeek PEAK hours: this costs about 2× the off-peak price. "
            f"Off-peak starts {status.next_change:%a %H:%M UTC}.[/yellow]"
        )

    router = Router(load_classifier(), load_routing_config(catalog), catalog)
    estimate = estimate_cost(prompts, router, catalog, offpeak_multiplier=catalog.price_schedule.off_peak_multiplier)
    console.print(
        f"\n[bold]Estimate for {estimate.prompts} prompts[/bold] (predicted routing {estimate.routed_tiers}; "
        f"every downgraded answer verified):\n"
        f"  answering ≈ ${estimate.serve_usd:.2f} · verification ≈ ${estimate.verification_usd:.2f} · "
        f"[bold]total ≈ ${estimate.total_usd:.2f}[/bold] at peak prices (≈ ${estimate.offpeak_total_usd:.2f} off-peak)"
    )
    if not args.yes and not confirm("Run the load test?"):
        console.print("Cancelled — nothing was sent.")
        return 1

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(base_url=args.api_url, headers=headers, timeout=timeout, transport=transport) as client:
        try:
            health = await check_api(client)
        except LoadtestAbort as error:
            console.print(f"[red]{error}[/red]")
            return 2

        started_at = now_utc()
        started = time.perf_counter()
        with Progress(TextColumn("Requests"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                      console=console) as progress:
            task = progress.add_task("send", total=len(prompts))
            results = await send_all(client, prompts, concurrency=args.concurrency,
                                     on_done=lambda _r: progress.advance(task))
        send_seconds = time.perf_counter() - started

        console.print("Waiting for the worker to verify every downgraded answer...")
        wait_started = time.perf_counter()
        drained, queue = await wait_for_verification(
            client, timeout_s=args.wait_timeout_min * 60,
            on_poll=lambda q: console.print(f"  queue: {q}", style="dim"),
        )
        if not drained:
            console.print(f"[yellow]Verification not finished after {args.wait_timeout_min} min; "
                          f"reporting what is done (pending ones are listed in the report).[/yellow]")
        stats = await fetch_stats(client, started_at)

    report = build_report(
        results=results,
        stats=stats,
        estimate=estimate,
        offpeak_multiplier=catalog.price_schedule.off_peak_multiplier,
        production_sample_rate=load_settings().verification.sample_rate_normal,
        meta={
            "generated_at": now_utc().isoformat(),
            "api_url": args.api_url,
            "classifier_version": health.get("classifier_version"),
            "mode": health.get("mode"),
            "escalation_mode": health.get("escalation_mode"),
            "pricing_status_at_start": status.describe(),
            "run": {
                "started_at": started_at.isoformat(),
                "duration_s": round(send_seconds, 1),
                "verification_wait_s": round(time.perf_counter() - wait_started, 1),
                "verification_drained": drained,
                "queue_at_end": queue,
                "concurrency": args.concurrency,
                "offset": args.offset,
            },
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    markdown = render_markdown(report)
    args.md.write_text(markdown, encoding="utf-8")
    console.print(Markdown(markdown))
    console.print(f"Saved → {args.out} and {args.md}")
    return 0 if report["run"]["failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
