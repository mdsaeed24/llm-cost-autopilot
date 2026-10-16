"""Background verification worker (run as its own process, next to the API).

    python scripts/worker.py            # run until Ctrl+C / SIGTERM
    python scripts/worker.py --once     # drain the queue, then exit
    python scripts/worker.py --limit 20 # process at most 20 jobs
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from rich.console import Console

from autopilot.bootstrap import build_worker
from autopilot.settings import load_settings

console = Console()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Exit when the queue is empty.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N jobs.")
    parser.add_argument("--concurrency", type=int, default=None, help="Jobs verified in parallel.")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    if args.concurrency:
        verification = settings.verification.model_copy(update={"worker_concurrency": args.concurrency})
        settings = settings.model_copy(update={"verification": verification})
    worker = build_worker(settings)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)  # finish the current batch, then exit

    console.print(f"Worker {worker.worker_id} started · queue: {worker.repo.queue_counts()}")
    try:
        stats = await worker.run(once=args.once, limit=args.limit, stop=stop)
    finally:
        await worker.verifier.client.aclose()
    console.print(
        f"Processed {stats.processed}: passed {stats.passed}, failed quality {stats.failed_quality}, "
        f"excluded {stats.excluded}, errors {stats.errors} · verification cost ${stats.cost_usd:.4f}"
    )
    for reason in stats.reasons:
        console.print(f"  [red]✗[/red] {reason}", markup=True)
    console.print(f"Queue now: {worker.repo.queue_counts()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
