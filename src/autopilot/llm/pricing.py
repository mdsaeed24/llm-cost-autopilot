"""Turn token counts into dollars, and tell whether DeepSeek is in its peak-price window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from autopilot.llm.schemas import ModelPricing, PriceSchedule

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_PER_MILLION = 1_000_000


def compute_cost(
    pricing: ModelPricing,
    *,
    cache_miss_tokens: int,
    cache_hit_tokens: int,
    output_tokens: int,
) -> float:
    """Cost in USD of one call, given how its prompt tokens split between cache hits and misses."""
    return (
        cache_miss_tokens * pricing.input_cache_miss_per_1m
        + cache_hit_tokens * pricing.input_cache_hit_per_1m
        + output_tokens * pricing.output_per_1m
    ) / _PER_MILLION


@dataclass(frozen=True)
class PeakStatus:
    """Whether ``at`` falls in a peak window, and when that changes next."""

    at: datetime
    is_peak: bool
    price_multiplier: float
    next_change: datetime | None

    def describe(self) -> str:
        """Human-readable one-liner for CLI output."""
        label = "PEAK" if self.is_peak else "OFF-PEAK"
        when = self.at.strftime("%a %H:%M UTC")
        if self.next_change is None:
            return f"DeepSeek pricing: {label} ({when})"
        delta = self.next_change - self.at
        hours, minutes = divmod(int(delta.total_seconds() // 60), 60)
        switch = "off-peak" if self.is_peak else "peak"
        return (
            f"DeepSeek pricing: {label} ({when}); switches to {switch} in {hours}h{minutes:02d}m "
            f"at {self.next_change.strftime('%a %H:%M UTC')}"
        )


def _parse_window(window: str) -> tuple[time, time | None]:
    start, end = window.split("-")
    start_t = time.fromisoformat(start)
    end_t = None if end == "24:00" else time.fromisoformat(end)  # None = midnight at day end
    return start_t, end_t


def is_peak(schedule: PriceSchedule, at: datetime) -> bool:
    """True if ``at`` (timezone-aware) is inside a peak window."""
    at = at.astimezone(UTC)
    if _WEEKDAYS[at.weekday()] not in schedule.peak_weekdays:
        return False
    now = at.time()
    for window in schedule.peak_windows:
        start, end = _parse_window(window)
        if start <= now and (end is None or now < end):
            return True
    return False


def peak_status(schedule: PriceSchedule, at: datetime | None = None) -> PeakStatus:
    """Peak/off-peak status at ``at`` (default: now), plus the next switch time within 8 days."""
    at = (at or datetime.now(UTC)).astimezone(UTC)
    current = is_peak(schedule, at)

    boundaries: list[datetime] = []
    for day_offset in range(8):
        day = (at + timedelta(days=day_offset)).date()
        midnight = datetime.combine(day, time(0), tzinfo=UTC)
        for window in schedule.peak_windows:
            start, end = _parse_window(window)
            boundaries.append(datetime.combine(day, start, tzinfo=UTC))
            boundaries.append(midnight + timedelta(days=1) if end is None else datetime.combine(day, end, tzinfo=UTC))
        boundaries.append(midnight)  # weekday changes happen at midnight

    next_change = next(
        (b for b in sorted(boundaries) if b > at and is_peak(schedule, b) != current),
        None,
    )
    multiplier = 1.0 if current else schedule.off_peak_multiplier
    return PeakStatus(at=at, is_peak=current, price_multiplier=multiplier, next_change=next_change)
