"""The promotion rule: a new classifier replaces the current one only if it is not worse.

Both models are scored on the same frozen test prompts with the same (latest) labels.
"Not worse" means Tier-3 recall AND accuracy are each at least the current model's, minus a
small tolerance. Tier-3 recall is checked first because under-routing hard prompts is the
costliest mistake a router can make.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from autopilot.classifier.metrics import EvalMetrics, evaluate
from autopilot.classifier.predict import TierPredictor


class PromotionDecision(BaseModel):
    """Whether to promote, why, and the numbers behind it."""

    model_config = ConfigDict(extra="forbid")

    promote: bool
    reason: str
    candidate: EvalMetrics
    current: EvalMetrics | None


def score_on(predictor: TierPredictor, texts: Sequence[str], tiers: Sequence[int]) -> EvalMetrics:
    """Score any predictor (e.g. the current production model) on the given labeled prompts."""
    return evaluate(list(tiers), [predictor.predict(t).tier for t in texts])


def decide_promotion(
    candidate: EvalMetrics, current: EvalMetrics | None, *, tolerance: float = 0.0
) -> PromotionDecision:
    """Apply the not-worse rule."""
    if current is None:
        return PromotionDecision(promote=True, reason="no current model", candidate=candidate, current=None)
    if candidate.tier3_recall < current.tier3_recall - tolerance:
        reason = f"Tier-3 recall dropped ({candidate.tier3_recall:.0%} < {current.tier3_recall:.0%})"
        return PromotionDecision(promote=False, reason=reason, candidate=candidate, current=current)
    if candidate.accuracy < current.accuracy - tolerance:
        reason = f"accuracy dropped ({candidate.accuracy:.0%} < {current.accuracy:.0%})"
        return PromotionDecision(promote=False, reason=reason, candidate=candidate, current=current)
    reason = (
        f"not worse: Tier-3 recall {current.tier3_recall:.0%} → {candidate.tier3_recall:.0%}, "
        f"accuracy {current.accuracy:.0%} → {candidate.accuracy:.0%}"
    )
    return PromotionDecision(promote=True, reason=reason, candidate=candidate, current=current)
