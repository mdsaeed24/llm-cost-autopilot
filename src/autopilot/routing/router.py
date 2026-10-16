"""Decide which model should answer a prompt."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from autopilot.classifier.predict import Prediction, TierPredictor
from autopilot.dataset.schemas import TIERS, TaskType, Tier
from autopilot.llm.catalog import ModelCatalog
from autopilot.llm.schemas import ModelConfig
from autopilot.routing.config import RoutingConfig

logger = logging.getLogger(__name__)

TOP_TIER: Tier = max(TIERS)


class RoutingDecision(BaseModel):
    """What the router decided and why — returned to API callers and logged."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    predicted_tier: Tier | None  # None when the classifier failed
    confidence: float | None
    probabilities: dict[int, float]
    routed_tier: Tier
    routed_up: bool
    model_name: str
    task_type: TaskType | None
    task_confidence: float | None
    classifier_version: str
    reason: str
    classifier_error: str | None = None


class Router:
    """classify → (route up if unsure) → look up the tier's model."""

    def __init__(self, classifier: TierPredictor, config: RoutingConfig, catalog: ModelCatalog) -> None:
        config.check_models_exist(catalog)
        self.classifier = classifier
        self.config = config
        self.catalog = catalog

    def decide(self, prompt: str) -> RoutingDecision:
        try:
            prediction = self.classifier.predict(prompt)
        except Exception as error:  # noqa: BLE001 - routing must never take a request down
            return self._fallback(error)
        routed_tier, reason = self._apply_confidence_rule(prediction)
        return RoutingDecision(
            predicted_tier=prediction.tier,
            confidence=round(prediction.confidence, 4),
            probabilities=prediction.probabilities,
            routed_tier=routed_tier,
            routed_up=routed_tier != prediction.tier,
            model_name=self.config.tiers[routed_tier],
            task_type=prediction.task_type,
            task_confidence=round(prediction.task_confidence, 4),
            classifier_version=self.classifier.version,
            reason=reason,
        )

    def _fallback(self, error: Exception) -> RoutingDecision:
        """Classifier unavailable (e.g. embedding model down): use the safe fallback tier."""
        tier = self.config.fallback_tier
        logger.warning("Classifier failed (%s: %s); routing to fallback tier %d", type(error).__name__, error, tier)
        return RoutingDecision(
            predicted_tier=None,
            confidence=None,
            probabilities={},
            routed_tier=tier,
            routed_up=False,
            model_name=self.config.tiers[tier],
            task_type=None,
            task_confidence=None,
            classifier_version=self.classifier.version,
            reason=f"classifier unavailable ({type(error).__name__}) → fallback tier {tier}",
            classifier_error=f"{type(error).__name__}: {error}",
        )

    def model_for(self, decision: RoutingDecision) -> ModelConfig:
        return self.catalog.get(decision.model_name)

    def _apply_confidence_rule(self, prediction: Prediction) -> tuple[Tier, str]:
        rule = self.config.confidence
        base = f"classifier predicted tier {prediction.tier} (p={prediction.confidence:.2f})"
        if not rule.enabled or prediction.confidence >= rule.threshold:
            return prediction.tier, base
        if prediction.tier == TOP_TIER:
            return prediction.tier, f"{base}; low confidence but already top tier"
        up: Tier = prediction.tier + 1  # type: ignore[assignment]
        return up, f"{base}; below confidence threshold {rule.threshold:.2f} → routed up to tier {up}"
