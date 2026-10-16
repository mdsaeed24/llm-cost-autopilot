"""Cost maths and the DeepSeek peak/off-peak schedule."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from autopilot.llm.catalog import load_catalog
from autopilot.llm.pricing import compute_cost, is_peak, peak_status
from autopilot.llm.schemas import ModelPricing, PriceSchedule

FLASH = ModelPricing(input_cache_miss_per_1m=0.30, input_cache_hit_per_1m=0.006, output_per_1m=1.20)


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


@pytest.fixture
def schedule() -> PriceSchedule:
    return load_catalog().price_schedule


def test_cost_splits_cache_hits_and_misses() -> None:
    cost = compute_cost(FLASH, cache_miss_tokens=1000, cache_hit_tokens=500, output_tokens=2000)
    assert cost == pytest.approx((1000 * 0.30 + 500 * 0.006 + 2000 * 1.20) / 1e6)


def test_free_model_costs_nothing() -> None:
    free = ModelPricing(input_cache_miss_per_1m=0, input_cache_hit_per_1m=0, output_per_1m=0)
    assert compute_cost(free, cache_miss_tokens=10_000, cache_hit_tokens=0, output_tokens=10_000) == 0


# 2026-01-05 is a Monday, 2026-01-09 a Friday, 2026-01-10 a Saturday.
@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (utc(2026, 1, 5, 0, 59), False),
        (utc(2026, 1, 5, 1, 0), True),   # window start is inclusive
        (utc(2026, 1, 5, 3, 59), True),
        (utc(2026, 1, 5, 4, 0), False),  # window end is exclusive
        (utc(2026, 1, 5, 5, 0), False),  # gap between windows
        (utc(2026, 1, 5, 6, 0), True),
        (utc(2026, 1, 5, 9, 59), True),
        (utc(2026, 1, 5, 10, 0), False),
        (utc(2026, 1, 9, 7, 0), True),   # Friday
        (utc(2026, 1, 10, 2, 0), False),  # Saturday is always off-peak
    ],
)
def test_is_peak(schedule: PriceSchedule, when: datetime, expected: bool) -> None:
    assert is_peak(schedule, when) is expected


def test_is_peak_converts_timezones(schedule: PriceSchedule) -> None:
    from zoneinfo import ZoneInfo

    ist = datetime(2026, 1, 5, 7, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # = 01:30 UTC Monday
    assert is_peak(schedule, ist)


@pytest.mark.parametrize(
    ("when", "next_change"),
    [
        (utc(2026, 1, 5, 2, 30), utc(2026, 1, 5, 4)),    # in peak → ends at 04:00
        (utc(2026, 1, 5, 4, 30), utc(2026, 1, 5, 6)),    # in gap → next peak 06:00
        (utc(2026, 1, 9, 11), utc(2026, 1, 12, 1)),      # Friday evening → Monday 01:00
        (utc(2026, 1, 10, 12), utc(2026, 1, 12, 1)),     # Saturday → Monday 01:00
    ],
)
def test_peak_status_next_change(schedule: PriceSchedule, when: datetime, next_change: datetime) -> None:
    status = peak_status(schedule, when)
    assert status.next_change == next_change


def test_peak_status_multiplier_and_text(schedule: PriceSchedule) -> None:
    peak = peak_status(schedule, utc(2026, 1, 5, 2))
    off = peak_status(schedule, utc(2026, 1, 10, 2))
    assert peak.is_peak and peak.price_multiplier == 1.0
    assert not off.is_peak and off.price_multiplier == 0.5
    assert "PEAK" in peak.describe() and "switches to off-peak in 2h00m" in peak.describe()
    assert off.describe().startswith("DeepSeek pricing: OFF-PEAK")


@pytest.mark.parametrize("window", ["1:00-4:00", "04:00-01:00", "25:00-26:00"])
def test_bad_windows_rejected(window: str) -> None:
    with pytest.raises(ValidationError):
        PriceSchedule(provider="deepseek", peak_weekdays=["mon"], peak_windows=[window], off_peak_multiplier=0.5)
