"""Routing-aware metrics and the route-up rule."""

from __future__ import annotations

import numpy as np
import pytest

from autopilot.classifier.metrics import apply_route_up, evaluate, threshold_analysis


def test_evaluate_counts_under_and_over_routing() -> None:
    true = [1, 1, 2, 3, 3, 3]
    pred = [1, 2, 2, 3, 1, 2]
    m = evaluate(true, pred)
    assert m.accuracy == pytest.approx(3 / 6)
    assert m.under_routing_rate == pytest.approx(2 / 6)  # 3→1, 3→2
    assert m.over_routing_rate == pytest.approx(1 / 6)   # 1→2
    assert m.severe_under_routing == 1
    assert m.tier3_recall == pytest.approx(1 / 3)
    assert m.per_tier[3].precision == pytest.approx(1.0)
    assert m.confusion_matrix == [[1, 1, 0], [0, 1, 0], [1, 1, 1]]


def test_evaluate_handles_missing_tier() -> None:
    m = evaluate([1, 2], [1, 2])
    assert m.per_tier[3].support == 0 and m.per_tier[3].recall == 0


PROBA = np.array(
    [
        [0.9, 0.05, 0.05],  # confident tier 1
        [0.45, 0.35, 0.2],  # unsure tier 1 → tier 2
        [0.2, 0.5, 0.3],    # unsure-ish tier 2
        [0.1, 0.4, 0.5],    # unsure tier 3 → stays 3
    ]
)


def test_route_up_bumps_only_low_confidence_and_caps_at_3() -> None:
    routed, bumped = apply_route_up(PROBA, [1, 2, 3], threshold=0.55)
    assert routed.tolist() == [1, 2, 3, 3]
    assert bumped.tolist() == [False, True, True, False]


def test_threshold_zero_changes_nothing() -> None:
    routed, bumped = apply_route_up(PROBA, [1, 2, 3], threshold=0.0)
    assert routed.tolist() == [1, 1, 2, 3] and not bumped.any()


def test_threshold_analysis_trades_accuracy_for_safety() -> None:
    true = [1, 2, 3, 3]
    rows = threshold_analysis(true, PROBA, [1, 2, 3], [0.0, 0.55])
    off, on = rows
    assert off.routed_up_rate == 0
    assert on.routed_up_rate == pytest.approx(0.5)
    assert on.under_routing_rate <= off.under_routing_rate
    assert on.tier3_recall >= off.tier3_recall
    assert sum(on.tier_share.values()) == pytest.approx(1.0)
