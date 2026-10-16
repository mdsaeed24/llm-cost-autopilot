"""Evaluation metrics, with routing-specific ones next to the usual accuracy/precision/recall.

Direction matters in routing:
- **under-routing** (predicted tier < true tier): a hard prompt goes to a weak model → quality risk.
- **over-routing**  (predicted tier > true tier): an easy prompt goes to a strong model → wasted money.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

TIER_LABELS = (1, 2, 3)


class TierScores(BaseModel):
    """Precision / recall / F1 for one tier."""

    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float
    support: int


class EvalMetrics(BaseModel):
    """Everything we report for one set of predictions."""

    model_config = ConfigDict(extra="forbid")

    n: int
    accuracy: float
    macro_f1: float
    per_tier: dict[int, TierScores]
    confusion_matrix: list[list[int]]  # rows = true tier 1..3, columns = predicted tier 1..3
    under_routing_rate: float
    severe_under_routing: int  # true tier 3 predicted as tier 1 (two steps too low)
    over_routing_rate: float

    @property
    def tier3_recall(self) -> float:
        return self.per_tier[3].recall


def evaluate(y_true: Sequence[int] | NDArray[np.int_], y_pred: Sequence[int] | NDArray[np.int_]) -> EvalMetrics:
    """Compute :class:`EvalMetrics` for tier predictions."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=list(TIER_LABELS), zero_division=0
    )
    return EvalMetrics(
        n=len(true),
        accuracy=float(accuracy_score(true, pred)),
        macro_f1=float(f1_score(true, pred, labels=list(TIER_LABELS), average="macro", zero_division=0)),
        per_tier={
            tier: TierScores(precision=float(p), recall=float(r), f1=float(f), support=int(s))
            for tier, p, r, f, s in zip(TIER_LABELS, precision, recall, f1, support, strict=True)
        },
        confusion_matrix=confusion_matrix(true, pred, labels=list(TIER_LABELS)).tolist(),
        under_routing_rate=float(np.mean(pred < true)),
        severe_under_routing=int(np.sum((true == 3) & (pred == 1))),
        over_routing_rate=float(np.mean(pred > true)),
    )


def apply_route_up(
    proba: NDArray[np.float64], classes: Sequence[int], threshold: float
) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    """Predicted tier per row, bumped one tier up (max 3) when the top probability < ``threshold``.

    Returns (routed tiers, mask of rows that were routed up).
    """
    classes_arr = np.asarray(classes)
    predicted = classes_arr[np.argmax(proba, axis=1)]
    low_confidence = proba.max(axis=1) < threshold
    routed = np.where(low_confidence, np.minimum(predicted + 1, max(TIER_LABELS)), predicted)
    return routed, low_confidence & (predicted < max(TIER_LABELS))


class ThresholdRow(BaseModel):
    """Effect of one confidence threshold on out-of-fold predictions."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    routed_up_rate: float
    accuracy: float
    under_routing_rate: float
    over_routing_rate: float
    tier3_recall: float
    tier_share: dict[int, float]  # fraction of traffic ending up on each tier


def threshold_analysis(
    y_true: Sequence[int] | NDArray[np.int_],
    proba: NDArray[np.float64],
    classes: Sequence[int],
    thresholds: Sequence[float],
) -> list[ThresholdRow]:
    """How accuracy, under-/over-routing and traffic mix change as the route-up threshold rises."""
    true = np.asarray(y_true)
    rows: list[ThresholdRow] = []
    for threshold in thresholds:
        routed, bumped = apply_route_up(proba, classes, threshold)
        metrics = evaluate(true, routed)
        rows.append(
            ThresholdRow(
                threshold=threshold,
                routed_up_rate=float(bumped.mean()),
                accuracy=metrics.accuracy,
                under_routing_rate=metrics.under_routing_rate,
                over_routing_rate=metrics.over_routing_rate,
                tier3_recall=metrics.tier3_recall,
                tier_share={t: float(np.mean(routed == t)) for t in TIER_LABELS},
            )
        )
    return rows
