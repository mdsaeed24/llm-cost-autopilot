"""The not-worse promotion rule."""

from __future__ import annotations

from autopilot.classifier.metrics import evaluate
from autopilot.classifier.predict import Prediction
from autopilot.classifier.promotion import decide_promotion, score_on

GOOD = evaluate([1, 2, 3, 3], [1, 2, 3, 3])        # accuracy 100%, T3 recall 100%
T3_WORSE = evaluate([1, 2, 3, 3], [1, 2, 3, 2])    # T3 recall 50%
ACC_WORSE = evaluate([1, 2, 3, 3], [2, 2, 3, 3])   # T3 recall 100%, accuracy 75%


def test_no_current_model_promotes() -> None:
    assert decide_promotion(T3_WORSE, None).promote


def test_equal_or_better_promotes() -> None:
    assert decide_promotion(GOOD, GOOD).promote
    decision = decide_promotion(GOOD, T3_WORSE)
    assert decision.promote and "not worse" in decision.reason


def test_tier3_recall_drop_blocks_even_with_better_accuracy() -> None:
    decision = decide_promotion(T3_WORSE, ACC_WORSE)
    assert not decision.promote and "Tier-3 recall dropped" in decision.reason


def test_accuracy_drop_blocks() -> None:
    decision = decide_promotion(ACC_WORSE, GOOD)
    assert not decision.promote and "accuracy dropped" in decision.reason


def test_tolerance() -> None:
    assert decide_promotion(ACC_WORSE, GOOD, tolerance=0.3).promote


class Always:
    version = "x"

    def __init__(self, tier: int) -> None:
        self.tier = tier

    def predict(self, prompt: str) -> Prediction:
        return Prediction(tier=self.tier, confidence=1.0, probabilities={self.tier: 1.0},  # type: ignore[arg-type]
                          task_type="qa_classification", task_confidence=1.0)


def test_score_on_uses_predictor() -> None:
    metrics = score_on(Always(3), ["a", "b"], [3, 1])
    assert metrics.accuracy == 0.5 and metrics.tier3_recall == 1.0
